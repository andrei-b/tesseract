/**********************************************************************
 * File:        ocrrow.cpp  (Formerly row.c)
 * Description: Code for the ROW class.
 * Author:      Ray Smith
 *
 * (C) Copyright 1991, Hewlett-Packard Ltd.
 ** Licensed under the Apache License, Version 2.0 (the "License");
 ** you may not use this file except in compliance with the License.
 ** You may obtain a copy of the License at
 ** http://www.apache.org/licenses/LICENSE-2.0
 ** Unless required by applicable law or agreed to in writing, software
 ** distributed under the License is distributed on an "AS IS" BASIS,
 ** WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 ** See the License for the specific language governing permissions and
 ** limitations under the License.
 *
 **********************************************************************/

// Include automatically generated configuration file if running autoconf.
#ifdef HAVE_CONFIG_H
#  include "config_auto.h"
#endif

#include "blobbox.h"
#include "ocrrow.h"

#include <algorithm>
#include <limits>

namespace tesseract {

void CollectScriptBlobs(WERD *word, std::vector<TBOX> *boxes) {
  for (auto *list : {word->cblob_list(), word->rej_cblob_list()}) {
    C_BLOB_IT it(list);
    for (it.mark_cycle_pt(); !it.cycled_list(); it.forward()) {
      const TBOX box = it.data()->bounding_box();
      if (box.width() >= 2 && box.height() >= 2) {
        boxes->push_back(box);
      }
    }
  }
}

std::vector<SubscriptPair> FindSubscriptPairs(std::vector<TBOX> boxes) {
  std::sort(boxes.begin(), boxes.end(), [](const TBOX &a, const TBOX &b) {
    return a.left() < b.left() || (a.left() == b.left() && a.height() > b.height());
  });
  // Join vertically separated pieces in the same columns, such as the dot of j.
  std::vector<TBOX> glyphs;
  for (const auto &box : boxes) {
    if (!glyphs.empty() && box.left() < glyphs.back().right() &&
        !box.y_overlap(glyphs.back()) &&
        std::min(box.height(), glyphs.back().height()) * 3 <=
            std::max(box.height(), glyphs.back().height())) {
      glyphs.back() += box;
    } else {
      glyphs.push_back(box);
    }
  }
  std::vector<SubscriptPair> pairs;
  for (size_t i = 1; i < glyphs.size(); ++i) {
    const TBOX &base = glyphs[i - 1], &index = glyphs[i];
    float h = base.height();
    int gap = index.left() - base.right();
    int drop = base.bottom() - index.bottom();
    if (h >= 9 && base.width() >= 0.22f * h && base.width() <= 1.6f * h &&
        index.width() >= 0.19f * h && index.width() <= 0.96f * h &&
        index.height() >= 0.37f * h && index.height() <= 0.95f * h &&
        gap >= -0.22f * h && gap <= 0.5f * h &&
        drop >= 0.10f * h && drop <= 0.68f * h && base.top() - index.top() >= 0.30f * h) {
      pairs.push_back({base, index, i > 1 ? glyphs[i - 2].right() : std::numeric_limits<int>::min(),
                       i + 1 < glyphs.size() ? glyphs[i + 1].left() : std::numeric_limits<int>::max()});
    }
  }
  return pairs;
}

bool ROW::HasRepeatedSubscripts() {
  std::vector<TBOX> boxes;
  WERD_IT it(word_list());
  for (it.mark_cycle_pt(); !it.cycled_list(); it.forward()) {
    CollectScriptBlobs(it.data(), &boxes);
  }
  auto pairs = FindSubscriptPairs(std::move(boxes));
  for (size_t i = 1; i < pairs.size(); ++i) {
    const TBOX &a = pairs[i - 1].base, &b = pairs[i].base;
    if (b.left() > pairs[i - 1].index.right() && b.height() >= 0.7f * a.height() &&
        b.height() <= 1.4f * a.height() && std::abs(b.bottom() - a.bottom()) <= 0.35f * a.height()) {
      return true;
    }
  }
  return false;
}

/**********************************************************************
 * ROW::ROW
 *
 * Constructor to build a ROW. Only the stats stuff are given here.
 * The words are added directly.
 **********************************************************************/
ROW::ROW(                // constructor
    int32_t spline_size, // no of segments
    int32_t *xstarts,    // segment boundaries
    double *coeffs,      // coefficients
    float x_height,      // line height
    float ascenders,     // ascender size
    float descenders,    // descender drop
    int16_t kern,        // char gap
    int16_t space        // word gap
    )
    : baseline(spline_size, xstarts, coeffs), para_(nullptr) {
  kerning = kern; // just store stuff
  spacing = space;
  xheight = x_height;
  ascrise = ascenders;
  bodysize = 0.0f;
  descdrop = descenders;
  has_drop_cap_ = false;
  lmargin_ = 0;
  rmargin_ = 0;
}

/**********************************************************************
 * ROW::ROW
 *
 * Constructor to build a ROW. Only the stats stuff are given here.
 * The words are added directly.
 **********************************************************************/

ROW::ROW(           // constructor
    TO_ROW *to_row, // source row
    int16_t kern,   // char gap
    int16_t space   // word gap
    )
    : para_(nullptr) {
  kerning = kern; // just store stuff
  spacing = space;
  xheight = to_row->xheight;
  bodysize = to_row->body_size;
  ascrise = to_row->ascrise;
  descdrop = to_row->descdrop;
  baseline = to_row->baseline;
  has_drop_cap_ = false;
  lmargin_ = 0;
  rmargin_ = 0;
}

// Returns the bounding box including the desired combination of upper and
// lower noise/diacritic elements.
TBOX ROW::restricted_bounding_box(bool upper_dots, bool lower_dots) const {
  TBOX box;
  // This is a read-only iteration of the words in the row.
  WERD_IT it(const_cast<WERD_LIST *>(&words));
  for (it.mark_cycle_pt(); !it.cycled_list(); it.forward()) {
    box += it.data()->restricted_bounding_box(upper_dots, lower_dots);
  }
  return box;
}

/**********************************************************************
 * ROW::recalc_bounding_box
 *
 * Set the bounding box correctly
 **********************************************************************/

void ROW::recalc_bounding_box() { // recalculate BB
  WERD *word;                     // current word
  WERD_IT it = &words;            // words of ROW
  int16_t left;                   // of word
  int16_t prev_left;              // old left

  if (!it.empty()) {
    word = it.data();
    prev_left = word->bounding_box().left();
    it.forward();
    while (!it.at_first()) {
      word = it.data();
      left = word->bounding_box().left();
      if (left < prev_left) {
        it.move_to_first();
        // words in BB order
        it.sort(word_comparator);
        break;
      }
      prev_left = left;
      it.forward();
    }
  }
  for (it.mark_cycle_pt(); !it.cycled_list(); it.forward()) {
    word = it.data();
    if (it.at_first()) {
      word->set_flag(W_BOL, true);
    } else {
      // not start of line
      word->set_flag(W_BOL, false);
    }
    if (it.at_last()) {
      word->set_flag(W_EOL, true);
    } else {
      // not end of line
      word->set_flag(W_EOL, false);
    }
    // extend BB as reqd
    bound_box += word->bounding_box();
  }
}

/**********************************************************************
 * ROW::move
 *
 * Reposition row by vector
 **********************************************************************/

void ROW::move(      // reposition row
    const ICOORD vec // by vector
) {
  WERD_IT it(&words); // word iterator

  for (it.mark_cycle_pt(); !it.cycled_list(); it.forward()) {
    it.data()->move(vec);
  }

  bound_box.move(vec);
  baseline.move(vec);
}

/**********************************************************************
 * ROW::print
 *
 * Display members
 **********************************************************************/

void ROW::print() const {
  tprintf("Kerning= %d\n", kerning);
  tprintf("Spacing= %d\n", spacing);
  bound_box.print();
  tprintf("Xheight= %f\n", xheight);
  tprintf("Ascrise= %f\n", ascrise);
  tprintf("Descdrop= %f\n", descdrop);
  tprintf("has_drop_cap= %d\n", has_drop_cap_);
  tprintf("lmargin= %d, rmargin= %d\n", lmargin_, rmargin_);
}

/**********************************************************************
 * ROW::plot
 *
 * Draw the ROW in the given colour.
 **********************************************************************/

#ifndef GRAPHICS_DISABLED
void ROW::plot(              // draw it
    ScrollView *window,      // window to draw in
    ScrollView::Color colour // colour to draw in
) {
  WERD *word;          // current word
  WERD_IT it = &words; // words of ROW

  for (it.mark_cycle_pt(); !it.cycled_list(); it.forward()) {
    word = it.data();
    word->plot(window, colour); // all in one colour
  }
}

/**********************************************************************
 * ROW::plot
 *
 * Draw the ROW in rainbow colours.
 **********************************************************************/

void ROW::plot(        // draw it
    ScrollView *window // window to draw in
) {
  WERD *word;          // current word
  WERD_IT it = &words; // words of ROW

  for (it.mark_cycle_pt(); !it.cycled_list(); it.forward()) {
    word = it.data();
    word->plot(window); // in rainbow colours
  }
}
#endif // !GRAPHICS_DISABLED

/**********************************************************************
 * ROW::operator=
 *
 * Assign rows by duplicating the row structure but NOT the WERDLIST
 **********************************************************************/

ROW &ROW::operator=(const ROW &source) {
  this->ELIST<ROW>::LINK::operator=(source);
  kerning = source.kerning;
  spacing = source.spacing;
  xheight = source.xheight;
  bodysize = source.bodysize;
  ascrise = source.ascrise;
  descdrop = source.descdrop;
  if (!words.empty()) {
    words.clear();
  }
  baseline = source.baseline; // QSPLINES must do =
  bound_box = source.bound_box;
  has_drop_cap_ = source.has_drop_cap_;
  lmargin_ = source.lmargin_;
  rmargin_ = source.rmargin_;
  para_ = source.para_;
  return *this;
}

} // namespace tesseract
