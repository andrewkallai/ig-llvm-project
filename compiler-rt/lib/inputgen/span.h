// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
//Implementation from llvm-project/libcxx/include/span

#include <array>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <limits>
#include <type_traits>

namespace __ig {

inline constexpr size_t dynamic_extent = std::numeric_limits<size_t>::max();

template <typename _Tp, size_t _Extent = dynamic_extent> class span {
public:
  // constants and types
  using element_type = _Tp;
  using value_type = std::remove_cv_t<_Tp>;
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using pointer = _Tp *;
  using const_pointer = const _Tp *;
  using reference = _Tp &;
  using const_reference = const _Tp &;

  static constexpr size_type extent = _Extent;

  // span constructors, copy, assignment, and destructor
  template <size_t _Sz = _Extent>
    requires(_Sz == 0)
  constexpr span() noexcept : data_{nullptr} {}

  constexpr span(const span &) noexcept = default;
  constexpr span &operator=(const span &other) noexcept = default;

  template <typename _It>
  constexpr explicit span(_It first, size_type count)
      : data_{std::to_address(first)} {
    (void)count;
    assert(_Extent == count &&
           "size mismatch in span's constructor (iterator, len)");
    assert((count == 0 || std::to_address(first) != nullptr) &&
           "passed nullptr with non-zero length in span's constructor "
           "(iterator, len)");
  }

  template <typename _It, typename _End>
  constexpr explicit span(_It first, _End last)
      : data_{std::to_address(first)} {
    // [span.cons]/10
    // Throws: When and what last - first throws.
    [[maybe_unused]] auto dist = last - first;
    assert(dist >= 0 &&
           "invalid range in span's constructor (iterator, sentinel)");
    assert(dist == _Extent && "invalid range in span's constructor (iterator, "
                              "sentinel): last - first != extent");
  }

  constexpr span(element_type (&arr)[_Extent]) noexcept : data_{arr} {}

  template <typename _OtherElementType>
    requires std::is_convertible_v<_OtherElementType (*)[], element_type (*)[]>
  constexpr span(std::array<_OtherElementType, _Extent> &arr) noexcept
      : data_{arr.data()} {}

  template <typename _OtherElementType>
    requires std::is_convertible_v<const _OtherElementType (*)[],
                                   element_type (*)[]>
  constexpr span(const std::array<_OtherElementType, _Extent> &arr) noexcept
      : data_{arr.data()} {}

  template <typename _Range>
    requires std::is_convertible_v<
        typename std::iterator_traits<typename _Range::iterator>::value_type (
                *)[],
        element_type (*)[]>
  constexpr explicit span(_Range &&r) : data_{std::data(r)} {
    assert(std::size(r) == _Extent &&
           "size mismatch in span's constructor (range)");
  }

  template <typename _OtherElementType>
    requires std::is_convertible_v<_OtherElementType (*)[], element_type (*)[]>
  constexpr span(const span<_OtherElementType, _Extent> &other) noexcept
      : data_{other.data()} {}

  template <typename _OtherElementType>
    requires std::is_convertible_v<_OtherElementType (*)[], element_type (*)[]>
  constexpr explicit span(
      const span<_OtherElementType, dynamic_extent> &other) noexcept
      : data_{other.data()} {
    assert(_Extent == other.size() &&
           "size mismatch in span's constructor (other span)");
  }

  // element access
  constexpr reference operator[](size_type idx) const noexcept {
    assert(idx < _Extent && "Index out of range");
    return data_[idx];
  }

  constexpr reference front() const noexcept {
    assert(_Extent > 0 && "front() on empty span");
    return data_[0];
  }

  constexpr reference back() const noexcept {
    assert(_Extent > 0 && "back() on empty span");
    return data_[_Extent - 1];
  }

  constexpr pointer data() const noexcept { return data_; }

  // observers
  constexpr size_type size() const noexcept { return _Extent; }
  constexpr size_type size_bytes() const noexcept {
    return _Extent * sizeof(element_type);
  }
  [[nodiscard]] constexpr bool empty() const noexcept { return _Extent == 0; }

  // subviews
  template <size_t _Count>
  constexpr span<element_type, _Count> first() const noexcept {
    static_assert(_Count <= _Extent, "first(_Count) out of range");
    return span<element_type, _Count>{data_, _Count};
  }

  template <size_t _Count>
  constexpr span<element_type, _Count> last() const noexcept {
    static_assert(_Count <= _Extent, "last(_Count) out of range");
    return span<element_type, _Count>{data_ + _Extent - _Count, _Count};
  }

  constexpr span<element_type, dynamic_extent>
  first(size_type count) const noexcept {
    assert(count <= _Extent && "first(count) out of range");
    return {data_, count};
  }

  constexpr span<element_type, dynamic_extent>
  last(size_type count) const noexcept {
    assert(count <= _Extent && "last(count) out of range");
    return {data_ + _Extent - count, count};
  }

  template <size_t _Offset, size_t _Count = dynamic_extent>
  constexpr auto subspan() const noexcept
      -> span<element_type,
              _Count != dynamic_extent ? _Count : _Extent - _Offset> {
    static_assert(_Offset <= _Extent,
                  "span<T, N>::subspan<Offset, Count>(): Offset out of range");
    static_assert(
        _Count == dynamic_extent || _Count <= _Extent - _Offset,
        "span<T, N>::subspan<Offset, Count>(): Offset + Count out of range");

    using _ReturnType =
        span<element_type,
             _Count != dynamic_extent ? _Count : _Extent - _Offset>;
    return _ReturnType{data() + _Offset,
                       _Count == dynamic_extent ? size() - _Offset : _Count};
  }

  constexpr span<element_type, dynamic_extent>
  subspan(size_type offset, size_type count = dynamic_extent) const noexcept {
    assert(offset <= _Extent &&
           "span<T, N>::subspan(offset, count): offset out of range");
    if (count == dynamic_extent)
      return {data() + offset, _Extent - offset};
    assert(count <= _Extent - offset &&
           "span<T, N>::subspan(offset, count): offset + count out of range");
    return {data() + offset, count};
  }

private:
  pointer data_;
};

template <typename _Tp> class span<_Tp, dynamic_extent> {
public:
  // constants and types
  using element_type = _Tp;
  using value_type = std::remove_cv_t<_Tp>;
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using pointer = _Tp *;
  using const_pointer = const _Tp *;
  using reference = _Tp &;
  using const_reference = const _Tp &;

  static constexpr size_type extent = dynamic_extent;

  // span constructors, copy, assignment, and destructor
  constexpr span() noexcept : data_{nullptr}, size_{0} {}

  template <typename _It>
  constexpr span(_It first, size_type count)
      : data_{std::to_address(first)}, size_{count} {
    assert((count == 0 || std::to_address(first) != nullptr) &&
           "passed nullptr with non-zero length in span's constructor "
           "(iterator, len)");
  }

  template <typename _It, typename _End,
            typename = std::enable_if_t<std::is_convertible_v<
                decltype(std::declval<_End>() - std::declval<_It>()),
                difference_type>>>
  constexpr explicit span(_It first, _End last)
      : data_{std::to_address(first)} {
    assert(last - first >= 0 &&
           "invalid range in span's constructor (iterator, sentinel)");
  }

  template <typename _OtherElementType, size_t _N>
    requires std::is_convertible_v<_OtherElementType (*)[], element_type (*)[]>
  constexpr span(std::array<_OtherElementType, _N> &arr) noexcept
      : data_{arr.data()}, size_{_N} {}

  template <typename _OtherElementType, size_t _N>
    requires std::is_convertible_v<const _OtherElementType (*)[],
                                   element_type (*)[]>
  constexpr span(const std::array<_OtherElementType, _N> &arr) noexcept
      : data_{arr.data()}, size_{_N} {}

  template <typename _Range>
    requires std::is_convertible_v<
                 typename std::iterator_traits<
                     typename _Range::iterator>::value_type (*)[],
                 element_type (*)[]>
  constexpr explicit span(_Range &&r)
      : data_{std::data(r)}, size_{std::size(r)} {}

  template <typename _OtherElementType, size_t _N>
    requires std::is_convertible_v<_OtherElementType (*)[], element_type (*)[]>
  constexpr span(const span<_OtherElementType, _N> &other) noexcept
      : data_{other.data()}, size_{other.size()} {}

  template <typename _OtherElementType>
    requires std::is_convertible_v<_OtherElementType (*)[], element_type (*)[]>
  constexpr span(const span<_OtherElementType, dynamic_extent> &other) noexcept
      : data_{other.data()}, size_{other.size()} {}

  // element access
  constexpr reference operator[](size_type idx) const noexcept {
    assert(idx < size_ && "Index out of range");
    return data_[idx];
  }

  constexpr reference front() const noexcept {
    assert(!empty() && "front() on empty span");
    return data_[0];
  }

  constexpr reference back() const noexcept {
    assert(!empty() && "back() on empty span");
    return data_[size_ - 1];
  }

  constexpr pointer data() const noexcept { return data_; }

  // observers
  constexpr size_type size() const noexcept { return size_; }
  constexpr size_type size_bytes() const noexcept {
    return size_ * sizeof(element_type);
  }
  [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }

  // subviews
  template <size_t _Count>
  constexpr span<element_type, _Count> first() const noexcept {
    assert(_Count <= size_ && "first(_Count) out of range");
    return span<element_type, _Count>{data_, _Count};
  }

  template <size_t _Count>
  constexpr span<element_type, _Count> last() const noexcept {
    assert(_Count <= size_ && "last(_Count) out of range");
    return span<element_type, _Count>{data_ + size_ - _Count, _Count};
  }

  constexpr span<element_type, dynamic_extent>
  first(size_type count) const noexcept {
    assert(count <= size_ && "first(count) out of range");
    return {data_, count};
  }

  constexpr span<element_type, dynamic_extent>
  last(size_type count) const noexcept {
    assert(count <= size_ && "last(count) out of range");
    return {data_ + size_ - count, count};
  }

  template <size_t _Offset, size_t _Count = dynamic_extent>
  constexpr auto subspan() const noexcept
      -> span<element_type,
              _Count != dynamic_extent ? _Count : dynamic_extent> {
    static_assert(_Offset <= size_, "span<T, dynamic_extent>::subspan<Offset, "
                                    "Count>(): Offset out of range");
    static_assert(_Count == dynamic_extent || _Count <= size_ - _Offset,
                  "span<T, dynamic_extent>::subspan<Offset, Count>(): Offset + "
                  "Count out of range");

    using _ReturnType =
        span<element_type, _Count != dynamic_extent ? _Count : dynamic_extent>;
    return _ReturnType{data() + _Offset,
                       _Count == dynamic_extent ? size() - _Offset : _Count};
  }

  constexpr span<element_type, dynamic_extent>
  subspan(size_type offset, size_type count = dynamic_extent) const noexcept {
    assert(
        offset <= size_ &&
        "span<T, dynamic_extent>::subspan(offset, count): offset out of range");
    if (count == dynamic_extent)
      return {data() + offset, size_ - offset};
    assert(count <= size_ - offset &&
           "span<T, dynamic_extent>::subspan(offset, count): offset + count "
           "out of range");
    return {data() + offset, count};
  }

private:
  pointer data_;
  size_type size_;
};

} // namespace __ig
