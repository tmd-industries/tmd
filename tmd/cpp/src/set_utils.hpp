// Copyright 2019-2025, Relay Therapeutics
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <numeric>
#include <set>
#include <vector>

namespace tmd {

template <typename T> std::set<T> unique_idxs(const std::vector<T> &idxs) {
  std::set<T> unique_idxs(idxs.begin(), idxs.end());
  if (unique_idxs.size() < idxs.size()) {
    throw std::runtime_error("atom indices must be unique");
  }
  return unique_idxs;
}

template <typename T> std::vector<T> set_to_vector(const std::set<T> &s) {
  std::vector<T> v(s.begin(), s.end());
  return v;
}

// Provided a number of indices and a subset of indices, construct
// the indices from the complete set of indices
template <typename T>
std::vector<T> get_indices_difference(const size_t N,
                                      const std::set<T> initial_idxs) {
  std::vector<T> all_idxs(N);
  std::iota(all_idxs.begin(), all_idxs.end(), 0);
  std::set<T> difference;
  std::set_difference(all_idxs.begin(), all_idxs.end(), initial_idxs.begin(),
                      initial_idxs.end(),
                      std::inserter(difference, difference.end()));

  std::vector<T> dif_vect(set_to_vector(difference));
  return dif_vect;
}

} // namespace tmd
