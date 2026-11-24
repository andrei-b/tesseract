// (C) Copyright 2026, Stefan Weil
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "include_gunit.h"

#include "boxword.h"
#include "ocrrow.h"
#include "ratngs.h"
#include "unicharset.h"

#include <cfloat>
#include <cmath>
#include <limits>

namespace tesseract {

// Regression test for tesseract-ocr/tesseract#4627.
//
// A word that the recognizer could not classify carries the "bad" certainty
// WERD_CHOICE::kBadCertainty. Textord::CleanupSingleRowResult() sums the
// certainties of the words in a row, and the per-word confidence is computed
// as 100 + 5 * certainty (AllWordConfidences, ChoiceIterator::Confidence).
// With the old sentinel of -FLT_MAX, a row holding two such words overflowed
// the float sum to -inf (and 5 * -FLT_MAX overflowed as well), which the
// tesseract CLI turns into a SIGFPE because main1() enables the FE_OVERFLOW
// trap, and which silently yields a -inf confidence for library users.
// kBadCertainty must therefore be finite *and* small enough in magnitude that
// these float sums and scalings cannot overflow.

class RatngsTest : public ::testing::Test {
protected:
  void SetUp() override {
    unicharset_.clear();
    unicharset_.unichar_insert(" ");
  }

  // The certainty a word is given when it could not be classified.
  float BadCertainty() const {
    return WERD_CHOICE::kBadCertainty;
  }

  UNICHARSET unicharset_;
};

// The sentinel itself must be a finite float; it is the value a missing
// classification is given (WERD_CHOICE::make_bad,
// WERD_RES::FakeWordFromRatings).
TEST_F(RatngsTest, BadCertaintyIsFinite) {
  const float bad = BadCertainty();
  EXPECT_TRUE(std::isfinite(bad));
  // It is a "bad" (very low) certainty, i.e. negative.
  EXPECT_LT(bad, 0.0f);
  EXPECT_GT(bad, std::numeric_limits<float>::lowest());
}

// make_bad() (and, by construction, FakeWordFromRatings which reuses the same
// constant) must leave a finite certainty that can be summed/scaled without
// overflowing the float arithmetic used downstream.
TEST_F(RatngsTest, BadWordCertaintyIsSummable) {
  WERD_CHOICE bad(&unicharset_);
  bad.make_bad();
  const float cert = bad.certainty();
  EXPECT_TRUE(std::isfinite(cert));
  EXPECT_FLOAT_EQ(WERD_CHOICE::kBadRating, bad.rating());

  // Two unclassified words in one row (Textord::CleanupSingleRowResult sums a
  // float per row). With the former -FLT_MAX sentinel this was -inf.
  EXPECT_TRUE(std::isfinite(cert + cert));
  EXPECT_FALSE(std::isinf(cert + cert));

  // Per-word confidence (AllWordConfidences / ChoiceIterator::Confidence).
  // With the former -FLT_MAX sentinel 5 * cert overflowed to -inf.
  EXPECT_TRUE(std::isfinite(100.0f + 5.0f * cert));
  EXPECT_FALSE(std::isinf(100.0f + 5.0f * cert));
}

// Original synthetic boxes, in Tesseract's bottom-up row coordinates.
TEST_F(RatngsTest, LSTMSubscriptGeometry) {
  for (const char *c : {"P", "n", "2", ","}) {
    unicharset_.unichar_insert(c);
  }
  unicharset_.set_isalpha(unicharset_.unichar_to_id("P"), true);
  unicharset_.set_isupper(unicharset_.unichar_to_id("P"), true);
  unicharset_.set_isalpha(unicharset_.unichar_to_id("n"), true);
  unicharset_.set_isdigit(unicharset_.unichar_to_id("2"), true);
  int32_t boundaries[] = {0, 200};
  double coeffs[] = {0.0, 0.0, 50.0};
  ROW row(1, boundaries, coeffs, 20, 8, -6, 2, 8);

  const TBOX base(5, 50, 25, 80);
  const TBOX index(26, 43, 39, 61);
  auto positions = [&](const char *text, std::initializer_list<TBOX> glyphs) {
    WERD_CHOICE word(text, unicharset_);
    BoxWord boxes;
    for (const TBOX &box : glyphs) {
      boxes.InsertBox(boxes.length(), box);
    }
    word.MarkLSTMSubscripts(boxes, row);
    std::string result;
    for (unsigned i = 0; i < word.length(); ++i) {
      result.push_back(word.BlobPosition(i) == SP_SUBSCRIPT ? '_' : '.');
    }
    return result;
  };
  EXPECT_EQ("._", positions("P2", {base, index}));
  EXPECT_EQ(".__", positions("Pnn", {base, index, TBOX(40, 43, 53, 61)}));
  EXPECT_EQ("..", positions("P2", {base, TBOX(26, 50, 39, 74)}));
  EXPECT_EQ("..", positions("P,", {base, TBOX(26, 45, 30, 54)}));
  EXPECT_EQ("....", positions("Pnnn", {base, index, index, index}));
  EXPECT_EQ("..", positions("P2", {base}));
}

// Connected components use bottom-up image coordinates. A dot and stem
// belong to the same index. On-baseline neighbors are ordinary letters.
TEST_F(RatngsTest, FindsLoweredInkBesideBase) {
  const TBOX base(10, 30, 30, 60), stem(31, 21, 39, 36), dot(34, 38, 37, 41);
  auto pairs = FindSubscriptPairs({dot, TBOX(66, 30, 78, 52), stem, base});
  ASSERT_EQ(1u, pairs.size());
  EXPECT_EQ(10, pairs[0].base.left());
  EXPECT_EQ(21, pairs[0].index.bottom());
  EXPECT_EQ(41, pairs[0].index.top());
  EXPECT_EQ(66, pairs[0].next_left);
  EXPECT_TRUE(FindSubscriptPairs({base, TBOX(31, 30, 42, 47)}).empty());
  EXPECT_TRUE(FindSubscriptPairs({base, TBOX(31, 27, 34, 32)}).empty());
}

} // namespace tesseract.
