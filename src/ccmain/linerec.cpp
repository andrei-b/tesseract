///////////////////////////////////////////////////////////////////////
// File:        linerec.cpp
// Description: Top-level line-based recognition module for Tesseract.
// Author:      Ray Smith
//
// (C) Copyright 2013, Google Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
///////////////////////////////////////////////////////////////////////

#include "tesseractclass.h"

#include "boxread.h"
#include "imagedata.h" // for ImageData
#include "lstmrecognizer.h"
#include "ocrrow.h"
#include "pageres.h"
#include "recodebeam.h"
#include "tprintf.h"

#include <algorithm>

namespace tesseract {

// Scale factor to make certainty more comparable to Tesseract.
const float kCertaintyScale = 7.0f;
// Worst acceptable certainty for a dictionary word.
const float kWorstDictCertainty = -25.0f;

// Generates training data for training a line recognizer, eg LSTM.
// Breaks the page into lines, according to the boxes, and writes them to a
// serialized DocumentData based on output_basename.
// Return true if successful, false if an error occurred.
bool Tesseract::TrainLineRecognizer(const char *input_imagename, const std::string &output_basename,
                                    BLOCK_LIST *block_list) {
  std::string lstmf_name = output_basename + ".lstmf";
  DocumentData images(lstmf_name);
  if (applybox_page > 0) {
    // Load existing document for the previous pages.
    if (!images.LoadDocument(lstmf_name.c_str(), 0, 0, nullptr)) {
      tprintf("Failed to read training data from %s!\n", lstmf_name.c_str());
      return false;
    }
  }
  std::vector<TBOX> boxes;
  std::vector<std::string> texts;
  // Get the boxes for this page, if there are any.
  if (!ReadAllBoxes(applybox_page, false, input_imagename, &boxes, &texts, nullptr, nullptr) ||
      boxes.empty()) {
    tprintf("Failed to read boxes from %s\n", input_imagename);
    return false;
  }
  TrainFromBoxes(boxes, texts, block_list, &images);
  if (images.PagesSize() == 0) {
    tprintf("Failed to read pages from %s\n", input_imagename);
    return false;
  }
  images.Shuffle();
  if (!images.SaveDocument(lstmf_name.c_str(), nullptr)) {
    tprintf("Failed to write training data to %s!\n", lstmf_name.c_str());
    return false;
  }
  return true;
}

// Generates training data for training a line recognizer, eg LSTM.
// Breaks the boxes into lines, normalizes them, converts to ImageData and
// appends them to the given training_data.
void Tesseract::TrainFromBoxes(const std::vector<TBOX> &boxes, const std::vector<std::string> &texts,
                               BLOCK_LIST *block_list, DocumentData *training_data) {
  auto box_count = boxes.size();
  // Process all the text lines in this page, as defined by the boxes.
  unsigned end_box = 0;
  // Don't let \t, which marks newlines in the box file, get into the line
  // content, as that makes the line unusable in training.
  while (end_box < texts.size() && texts[end_box] == "\t") {
    ++end_box;
  }
  for (auto start_box = end_box; start_box < box_count; start_box = end_box) {
    // Find the textline of boxes starting at start and their bounding box.
    TBOX line_box = boxes[start_box];
    std::string line_str = texts[start_box];
    for (end_box = start_box + 1; end_box < box_count && texts[end_box] != "\t"; ++end_box) {
      line_box += boxes[end_box];
      line_str += texts[end_box];
    }
    // Find the most overlapping block.
    BLOCK *best_block = nullptr;
    int best_overlap = 0;
    BLOCK_IT b_it(block_list);
    for (b_it.mark_cycle_pt(); !b_it.cycled_list(); b_it.forward()) {
      BLOCK *block = b_it.data();
      if (block->pdblk.poly_block() != nullptr && !block->pdblk.poly_block()->IsText()) {
        continue; // Not a text block.
      }
      TBOX block_box = block->pdblk.bounding_box();
      block_box.rotate(block->re_rotation());
      if (block_box.major_overlap(line_box)) {
        TBOX overlap_box = line_box.intersection(block_box);
        if (overlap_box.area() > best_overlap) {
          best_overlap = overlap_box.area();
          best_block = block;
        }
      }
    }
    ImageData *imagedata = nullptr;
    if (best_block == nullptr) {
      tprintf("No block overlapping textline: %s\n", line_str.c_str());
    } else {
      imagedata = GetLineData(line_box, boxes, texts, start_box, end_box, *best_block);
    }
    if (imagedata != nullptr) {
      training_data->AddPageToDocument(imagedata);
    }
    // Don't let \t, which marks newlines in the box file, get into the line
    // content, as that makes the line unusable in training.
    while (end_box < texts.size() && texts[end_box] == "\t") {
      ++end_box;
    }
  }
}

// Returns an Imagedata containing the image of the given box,
// and ground truth boxes/truth text if available in the input.
// The image is not normalized in any way.
ImageData *Tesseract::GetLineData(const TBOX &line_box, const std::vector<TBOX> &boxes,
                                  const std::vector<std::string> &texts, int start_box, int end_box,
                                  const BLOCK &block) {
  TBOX revised_box;
  ImageData *image_data = GetRectImage(line_box, block, kImagePadding, &revised_box);
  if (image_data == nullptr) {
    return nullptr;
  }
  image_data->set_page_number(applybox_page);
  // Copy the boxes and shift them so they are relative to the image.
  FCOORD block_rotation(block.re_rotation().x(), -block.re_rotation().y());
  ICOORD shift = -revised_box.botleft();
  std::vector<TBOX> line_boxes;
  std::vector<std::string> line_texts;
  for (int b = start_box; b < end_box; ++b) {
    TBOX box = boxes[b];
    box.rotate(block_rotation);
    box.move(shift);
    line_boxes.push_back(box);
    line_texts.push_back(texts[b]);
  }
  std::vector<int> page_numbers(line_boxes.size(), applybox_page);
  image_data->AddBoxes(line_boxes, line_texts, page_numbers);
  return image_data;
}

// Helper gets the image of a rectangle, using the block.re_rotation() if
// needed to get to the image, and rotating the result back to horizontal
// layout. (CJK characters will be on their left sides) The vertical text flag
// is set in the returned ImageData if the text was originally vertical, which
// can be used to invoke a different CJK recognition engine. The revised_box
// is also returned to enable calculation of output bounding boxes.
ImageData *Tesseract::GetRectImage(const TBOX &box, const BLOCK &block, int padding,
                                   TBOX *revised_box) const {
  TBOX wbox = box;
  wbox.pad(padding, padding);
  *revised_box = wbox;
  // Number of clockwise 90 degree rotations needed to get back to tesseract
  // coords from the clipped image.
  int num_rotations = 0;
  if (block.re_rotation().y() > 0.0f) {
    num_rotations = 1;
  } else if (block.re_rotation().x() < 0.0f) {
    num_rotations = 2;
  } else if (block.re_rotation().y() < 0.0f) {
    num_rotations = 3;
  }
  // Handle two cases automatically: 1 the box came from the block, 2 the box
  // came from a box file, and refers to the image, which the block may not.
  if (block.pdblk.bounding_box().major_overlap(*revised_box)) {
    revised_box->rotate(block.re_rotation());
  }
  // Now revised_box always refers to the image.
  // BestPix is never colormapped, but may be of any depth.
  Image pix = BestPix();
  int width = pixGetWidth(pix);
  int height = pixGetHeight(pix);
  TBOX image_box(0, 0, width, height);
  // Clip to image bounds;
  *revised_box &= image_box;
  if (revised_box->null_box()) {
    return nullptr;
  }
  Box *clip_box = boxCreate(revised_box->left(), height - revised_box->top(), revised_box->width(),
                            revised_box->height());
  Image box_pix = pixClipRectangle(pix, clip_box, nullptr);
  boxDestroy(&clip_box);
  if (box_pix == nullptr) {
    return nullptr;
  }
  if (num_rotations > 0) {
    Image rot_pix = pixRotateOrth(box_pix, num_rotations);
    box_pix.destroy();
    box_pix = rot_pix;
  }
  // Convert sub-8-bit images to 8 bit.
  int depth = pixGetDepth(box_pix);
  if (depth < 8) {
    Image grey;
    grey = pixConvertTo8(box_pix, false);
    box_pix.destroy();
    box_pix = grey;
  }
  bool vertical_text = false;
  if (num_rotations > 0) {
    // Rotated the clipped revised box back to internal coordinates.
    FCOORD rotation(block.re_rotation().x(), -block.re_rotation().y());
    revised_box->rotate(rotation);
    if (num_rotations != 2) {
      vertical_text = true;
    }
  }
  return new ImageData(vertical_text, box_pix);
}

// Extract one ink component from a borrowed page in bottom-up coordinates.
static Image ScaledGlyph(Image page, const TBOX &box, int height) {
  if (!page || !TBOX(0, 0, pixGetWidth(page), pixGetHeight(page)).contains(box)) {
    return nullptr;
  }
  Box *rect = boxCreate(box.left(), pixGetHeight(page) - box.top(), box.width(), box.height());
  Image glyph = pixClipRectangle(page, rect, nullptr);
  boxDestroy(&rect);
  if (!glyph) {
    return nullptr;
  }
  Image gray = pixConvertTo8(glyph, false);
  glyph.destroy();
  Image scaled = gray ? pixScaleToSize(gray, 0, height) : nullptr;
  gray.destroy();
  return scaled;
}

static Image BaselineAlignedPair(Image binary, const SubscriptPair &pair) {
  constexpr int kBody = 40, kIndex = 34, kPad = 5, kGap = 2;
  Image base = ScaledGlyph(binary, pair.base, kBody);
  Image index = ScaledGlyph(binary, pair.index, kIndex);
  Image canvas;
  if (base && index) {
    int next = kPad + pixGetWidth(base) + kGap;
    canvas = pixCreate(next + pixGetWidth(index) + kPad, kBody + 2 * kPad, 8);
    if (canvas) {
      pixSetAll(canvas);
      pixRasterop(canvas, kPad, kPad, pixGetWidth(base), kBody, PIX_SRC, base, 0, 0);
      pixRasterop(canvas, next, kPad + kBody - kIndex, pixGetWidth(index), kIndex,
                  PIX_SRC, index, 0, 0);
    }
  }
  base.destroy();
  index.destroy();
  return canvas;
}

// A row that nearly failed layout can have a bad fitted baseline. Two
// distinct decoder slots corroborated by the original ink need no reread.
static void MarkAlignedPair(WERD_RES *word, const SubscriptPair &pair, const Dict *dict) {
  WERD_CHOICE *choice = word->best_choice;
  if (!choice || choice->length() < 2 || choice->length() > 4) {
    return;
  }
  // In a word such as "By" the descender reaches below the baseline, but
  // the top of the y is still near x-height. Require deeper placement before
  // overriding a dictionary word (or when there is no language dictionary).
  if (pair.base.top() - pair.index.top() < 0.55f * pair.base.height() &&
      (!dict || dict->valid_word(*choice))) {
    return;
  }
  const UNICHARSET &cs = *word->uch_set;
  C_BLOB_IT it(word->word->cblob_list());
  if (it.length() != choice->length()) {
    return;
  }
  auto alnum = [&cs](UNICHAR_ID id) { return cs.get_isalpha(id) || cs.get_isdigit(id); };
  TBOX prev = it.data()->bounding_box();
  for (unsigned i = 1; i < choice->length(); ++i) {
    it.forward();
    TBOX current = it.data()->bounding_box();
    auto base = choice->unichar_id(i - 1), index = choice->unichar_id(i);
    if (cs.get_isalpha(base) && alnum(index) && (cs.get_isupper(base) || cs.get_isdigit(index)) &&
        (i == 1 || !alnum(choice->unichar_id(i - 2))) &&
        (i + 1 == choice->length() || !alnum(choice->unichar_id(i + 1))) &&
        prev.left() <= pair.base.x_middle() && pair.base.x_middle() < prev.right() &&
        current.left() <= pair.index.x_middle() && pair.index.x_middle() < current.right()) {
      choice->SetBlobPosition(i, SP_SUBSCRIPT);
      return;
    }
    prev = current;
  }
}

// Read the lowered ink without its base, preserving the decoder's real
// alternatives for callers that request symbol choices.
static UNICHAR_ID ReadIsolatedIndex(LSTMRecognizer *recognizer, Image page, const TBOX &box,
                                    int height, int choice_mode, int iterations,
                                    PointerVector<WERD_RES> *result) {
  Image glyph = ScaledGlyph(page, box, height);
  if (!glyph) {
    return INVALID_UNICHAR_ID;
  }
  Image padded = pixAddBorder(glyph, height / 4, 255);
  glyph.destroy();
  if (!padded) {
    return INVALID_UNICHAR_ID;
  }
  TBOX bounds(0, 0, pixGetWidth(padded), pixGetHeight(padded));
  ImageData sample(false, padded);
  recognizer->RecognizeLine(sample, 0, false, kWorstDictCertainty / kCertaintyScale, bounds,
                            result, choice_mode, iterations);
  if (result->size() != 1 || !(*result)[0]->best_choice ||
      (*result)[0]->best_choice->length() != 1) {
    return INVALID_UNICHAR_ID;
  }
  const auto &choice = *(*result)[0]->best_choice;
  const auto &cs = recognizer->GetUnicharset();
  auto id = choice.unichar_id(0);
  if (choice.certainty(0) < -0.8f || (!cs.get_isalpha(id) && !cs.get_isdigit(id))) {
    return INVALID_UNICHAR_ID;
  }
  return id;
}

static bool IsLoweredSlot(const SubscriptPair &pair, const ROW &row, const WERD_CHOICE &word,
                          const UNICHARSET &cs, const Dict *dict) {
  const auto index = word.unichar_id(1);
  if (cs.get_isalpha(index) && (!dict || dict->valid_word(word))) {
    return false;
  }
  const float h = pair.base.height();
  const auto &b = pair.index;
  // A comma can descend as far as a numeral. Require letter-sized ink
  // above the local base bottom before reinterpreting punctuation.
  // Ink edges are measured in whole pixels. Round the required overlap down
  // so an index on a 28-pixel capital can qualify with nine pixels of ink.
  if (cs.get_ispunctuation(index) && b.top() - pair.base.bottom() < pair.base.height() / 3) {
    return false;
  }
  return b.height() >= 0.45f * h && b.width() >= 0.22f * h &&
         row.base_line(b.x_middle()) - b.bottom() >= 0.07f * h &&
         pair.base.top() - b.top() >= 0.36f * h;
}

// LSTM words have one ratings cell per character. Transfer the isolated
// classification to that cell, then rebuild both the raw and cooked choices.
static void SetIndexFromGlyph(WERD_RES *glyph, int choice_mode, WERD_RES *word) {
  std::swap((*word->ratings)(1, 1), (*glyph->ratings)(0, 0));
  BLOB_CHOICE_IT(word->ratings->get(1, 1)).data()->set_matrix_cell(1, 1);
  word->ClearWordChoices();
  word->FakeWordFromRatings(TOP_CHOICE_PERM);
  if (!choice_mode) {
    return;
  }
  // Both inputs end at the glyph in question. CTC may include an initial
  // space, or omit choices at the end; preserve those offsets.
  glyph->CTC_symbol_choices.resize(glyph->end);
  word->CTC_symbol_choices.resize(word->end);
  word->CTC_symbol_choices.back() = std::move(glyph->CTC_symbol_choices.back());
  glyph->segmented_timesteps.resize(glyph->end);
  word->segmented_timesteps.resize(word->end);
  word->segmented_timesteps.back() = std::move(glyph->segmented_timesteps.back());
  word->timesteps.clear();
  for (const auto &segment : word->segmented_timesteps) {
    word->timesteps.insert(word->timesteps.end(), segment.begin(), segment.end());
  }
}

// A two-character word already has decoder boxes. Replace only the second
// classifier result, keeping the base, spacing and original geometry.
static void RereadLoweredSlot(LSTMRecognizer *recognizer, Image binary, Image original,
                              const SubscriptPair &pair, int choice_mode, int iterations,
                              WERD_RES *word) {
  C_BLOB_IT blobs(word->word->cblob_list());
  if (blobs.length() != 2) {
    return;
  }
  TBOX first = blobs.data()->bounding_box();
  blobs.forward();
  TBOX last = blobs.data()->bounding_box();
  if (first.left() > pair.base.x_middle() || pair.base.x_middle() >= first.right() ||
      last.left() > pair.index.x_middle() || pair.index.x_middle() >= last.right()) {
    return;
  }
  PointerVector<WERD_RES> binary_read, gray_read;
  auto id = ReadIsolatedIndex(recognizer, binary, pair.index, 48, 0, 0, &binary_read);
  if (id == INVALID_UNICHAR_ID ||
      ReadIsolatedIndex(recognizer, original, pair.index, 40, choice_mode, iterations,
                        &gray_read) != id) {
    return;
  }
  if (id != word->best_choice->unichar_id(1)) {
    // A first-pass digit is already a plausible index. Avoid exchanging one
    // plausible digit for a different reading on a smaller crop.
    if (word->uch_set->get_isdigit(word->best_choice->unichar_id(1))) {
      return;
    }
    SetIndexFromGlyph(gray_read[0], choice_mode, word);
  }
  word->best_choice->SetBlobPosition(1, SP_SUBSCRIPT);
}

// Use the decoder's two separate slots to find the original ink even
// when a digit next to an x-height base is too tall for FindSubscriptPairs.
static TBOX InkInsideSlot(const std::vector<TBOX> &ink, const TBOX &slot) {
  TBOX result;
  for (const auto &blob : ink) {
    if (slot.left() <= blob.x_middle() && blob.x_middle() < slot.right() &&
        slot.bottom() <= blob.y_middle() && blob.y_middle() < slot.top()) {
      result += blob;
    }
  }
  return result;
}

static void RereadTallPunctuationSlots(LSTMRecognizer *recognizer, Image binary, Image original,
                                       const std::vector<TBOX> &ink, int choice_mode, int iterations,
                                       PointerVector<WERD_RES> *words) {
  const UNICHARSET &charset = recognizer->GetUnicharset();
  const auto comma = charset.unichar_to_id(",");
  const auto semicolon = charset.unichar_to_id(";");
  int attempts = 0;
  for (size_t w = 0; w < words->size(); ++w) {
    WERD_RES *word = (*words)[w];
    const auto *choice = word->best_choice;
    if (!choice || choice->length() != 2 || !charset.get_isalpha(choice->unichar_id(0))) {
      continue;
    }
    const auto mark = choice->unichar_id(1);
    if (mark != comma && mark != semicolon) {
      continue;
    }
    C_BLOB_IT slots(word->word->cblob_list());
    if (slots.length() != 2) {
      continue;
    }
    const TBOX first = slots.data()->bounding_box();
    slots.forward();
    const TBOX second = slots.data()->bounding_box();
    const TBOX base = InkInsideSlot(ink, first);
    const TBOX index = InkInsideSlot(ink, second);
    if (base.null_box() || index.null_box()) {
      continue;
    }
    const float h = base.height();
    const int drop = base.bottom() - index.bottom();
    const int gap = index.left() - base.right();
    if (h < 9 || index.height() < 0.72f * h || index.height() > 1.40f * h ||
        index.width() < 0.20f * h || index.width() > h ||
        base.top() - index.top() < 0.15f * h || index.top() - base.bottom() < 0.40f * h ||
        drop < 0.15f * h || drop > 0.75f * h || gap < 0 || gap > 0.43f * h) {
      continue;
    }
    RereadLoweredSlot(recognizer, binary, original, {base, index, 0, 0}, choice_mode, iterations, word);
    if (++attempts >= 8) {
      return;
    }
  }
}

// Existing decoder slots keep their original segmentation. Missing or split
// slots use the separate base-and-index recovery path.
static void RecoverSubscriptWords(LSTMRecognizer *recognizer, Image binary, Image original,
                                  WERD *input, const ROW &row, bool recover, bool reread,
                                  int choice_mode, int iterations, PointerVector<WERD_RES> *words) {
  std::vector<TBOX> boxes;
  CollectScriptBlobs(input, &boxes);
  const auto pairs = FindSubscriptPairs(boxes);
  const UNICHARSET &charset = recognizer->GetUnicharset();
  size_t w = 0;
  int recovery_tries = 0, slot_tries = 0;
  for (const auto &pair : pairs) {
    const TBOX &base = pair.base, &index = pair.index;
    while (w < words->size() && (*words)[w]->word->bounding_box().right() <= base.x_middle()) {
      ++w;
    }
    if (w == words->size()) {
      break;
    }
    WERD_RES *old = (*words)[w];
    if (recover) {
      MarkAlignedPair(old, pair, recognizer->GetDict());
    }
    int clearance = std::max(2, base.height() / 7);
    if (base.right() > index.left() || pair.previous_right > base.left() - clearance ||
        pair.next_left < index.right() + clearance) {
      continue;
    }
    TBOX bounds = old->word->bounding_box();
    if (bounds.left() > base.x_middle() || bounds.right() < index.x_middle() ||
        bounds.left() < pair.previous_right || bounds.right() > pair.next_left) {
      continue;
    }
    WERD_CHOICE *first = old->best_choice;
    if (!first || first->empty() || !charset.get_isalpha(first->unichar_id(0))) {
      continue;
    }
    if (first->length() == 2) {
      if (reread && slot_tries < 8 &&
          IsLoweredSlot(pair, row, *first, charset, recognizer->GetDict())) {
        ++slot_tries;
        RereadLoweredSlot(recognizer, binary, original, pair, choice_mode, iterations, old);
      }
      continue;
    }
    if (!recover || recovery_tries >= 8 || (first->length() != 1 && first->length() != 3)) {
      continue;
    }
    if (first->length() == 3 && (!charset.get_ispunctuation(first->unichar_id(1)) ||
                                 !charset.get_ispunctuation(first->unichar_id(2)))) {
      continue;
    }

    Image aligned = BaselineAlignedPair(binary, pair);
    if (!aligned) {
      continue;
    }
    ++recovery_tries;
    TBOX synthetic(0, 0, pixGetWidth(aligned), pixGetHeight(aligned));
    ImageData sample(false, aligned); // takes ownership
    PointerVector<WERD_RES> retry;
    recognizer->RecognizeLine(sample, 0, false, kWorstDictCertainty / kCertaintyScale,
                              synthetic, &retry, choice_mode, iterations);
    if (retry.size() != 1 || !retry[0]->best_choice) {
      continue;
    }
    WERD_CHOICE *second = retry[0]->best_choice;
    if (second->length() != 2 || second->unichar_id(0) != first->unichar_id(0) ||
        (!charset.get_isalpha(second->unichar_id(1)) && !charset.get_isdigit(second->unichar_id(1))) ||
        second->certainty(1) < -1.0f) {
      continue;
    }
    second->SetBlobPosition(1, SP_SUBSCRIPT);
    retry[0]->word->cblob_list()->clear();
    C_BLOB_IT blobs(retry[0]->word->cblob_list());
    blobs.add_after_then_move(C_BLOB::FakeBlob(base));
    blobs.add_after_then_move(C_BLOB::FakeBlob(index));
    retry[0]->word->set_blanks(old->word->space());
    retry[0]->space_certainty = old->space_certainty;
    std::swap((*words)[w], retry[0]);
    ++w;
  }
  if (reread) {
    RereadTallPunctuationSlots(recognizer, binary, original, boxes, choice_mode, iterations, words);
  }
}

// Recognizes a word or group of words, converting to WERD_RES in *words.
// Analogous to classify_word_pass1, but can handle a group of words as well.
void Tesseract::LSTMRecognizeWord(const BLOCK &block, ROW *row, WERD_RES *word,
                                  PointerVector<WERD_RES> *words) {
  TBOX word_box = word->word->bounding_box();
  // Get the word image - no frills.
  if (tessedit_pageseg_mode == PSM_SINGLE_WORD || tessedit_pageseg_mode == PSM_RAW_LINE) {
    // In single word mode, use the whole image without any other row/word
    // interpretation.
    word_box = TBOX(0, 0, ImageWidth(), ImageHeight());
  } else {
    float baseline = row->base_line((word_box.left() + word_box.right()) / 2);
    if (baseline + row->descenders() < word_box.bottom()) {
      word_box.set_bottom(baseline + row->descenders());
    }
    if (baseline + row->x_height() + row->ascenders() > word_box.top()) {
      word_box.set_top(baseline + row->x_height() + row->ascenders());
    }
  }
  ImageData *im_data = GetRectImage(word_box, block, kImagePadding, &word_box);
  if (im_data == nullptr) {
    return;
  }

  bool do_invert = tessedit_do_invert;
  float threshold = do_invert ? double(invert_threshold) : 0.0f;
  lstm_recognizer_->RecognizeLine(*im_data, threshold, classify_debug_level > 0,
                                  kWorstDictCertainty / kCertaintyScale, word_box, words,
                                  lstm_choice_mode, lstm_choice_iterations);
  delete im_data;
  if ((lstm_recover_subscripts || lstm_reread_subscripts) && SupportsSubscriptRecovery() &&
      !words->empty() && block.re_rotation().x() == 1.0f && block.classify_rotation().x() == 1.0f) {
    RecoverSubscriptWords(lstm_recognizer_, pix_binary_, BestPix(), word->word, *row,
                          lstm_recover_subscripts, lstm_reread_subscripts, lstm_choice_mode,
                          lstm_choice_iterations, words);
  }
  SearchWords(words);
}

// Apply segmentation search to the given set of words, within the constraints
// of the existing ratings matrix. If there is already a best_choice on a word
// leaves it untouched and just sets the done/accepted etc flags.
void Tesseract::SearchWords(PointerVector<WERD_RES> *words) {
  // Run the segmentation search on the network outputs and make a BoxWord
  // for each of the output words.
  // If we drop a word as junk, then there is always a space in front of the
  // next.
  const Dict *stopper_dict = lstm_recognizer_->GetDict();
  if (stopper_dict == nullptr) {
    stopper_dict = &getDict();
  }
  for (unsigned w = 0; w < words->size(); ++w) {
    WERD_RES *word = (*words)[w];
    if (word->best_choice == nullptr) {
      // It is a dud.
      word->SetupFake(lstm_recognizer_->GetUnicharset());
    } else {
      // Set the best state.
      for (unsigned i = 0; i < word->best_choice->length(); ++i) {
        int length = word->best_choice->state(i);
        word->best_state.push_back(length);
      }
      word->reject_map.initialise(word->best_choice->length());
      word->tess_failed = false;
      word->tess_accepted = true;
      word->tess_would_adapt = false;
      word->done = true;
      word->tesseract = this;
      float word_certainty = std::min(word->space_certainty, word->best_choice->certainty());
      word_certainty *= kCertaintyScale;
      if (getDict().stopper_debug_level >= 1) {
        tprintf("Best choice certainty=%g, space=%g, scaled=%g, final=%g\n",
                word->best_choice->certainty(), word->space_certainty,
                std::min(word->space_certainty, word->best_choice->certainty()) * kCertaintyScale,
                word_certainty);
        word->best_choice->print();
      }
      word->best_choice->set_certainty(word_certainty);

      word->tess_accepted = stopper_dict->AcceptableResult(word);
    }
  }
}

} // namespace tesseract.
