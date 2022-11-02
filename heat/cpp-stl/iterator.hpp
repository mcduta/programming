/*

  Blog on C++ STL with NVidia:
    https://developer.nvidia.com/blog/accelerating-standard-c-with-gpus-using-stdpar/

  This header file effectively contains the
    struct counting_iterator

  from the LULESH source (https://github.com/LLNL/LULESH.git), namely the file
    stdpar/src/lulesh.h

 */

# include <iterator>

typedef int32_t index_t;

struct array_iterator {

private:
  using self = array_iterator;

public:
  using value_type = index_t;
  using pointer    = index_t*;
  using reference  = index_t&;
  using difference_type   = typename std::make_signed<index_t>::type;
  using iterator_category = std::random_access_iterator_tag;

  // contructor
  array_iterator () : value(0) { }
  explicit array_iterator (value_type v) : value(v) { }

  // reference
  value_type operator *() const { return value; }

  // index reference
  value_type operator [] (difference_type n) const { return value + n; }

  // prefix incrementor
  self& operator ++ ()    { ++value; return *this; }
  self  operator ++ (int) { self result{value}; ++value; return result; }

  // prefix decrementor
  self& operator -- ()    { --value; return *this; }
  self  operator -- (int) { self result{value}; --value; return result; }

  // += operator
  self& operator += (difference_type n) { value += n; return *this; }

  // -= operator
  self& operator -= (difference_type n) { value -= n; return *this; }

  // + operator
  friend self operator + (self const& i, difference_type n) { return self(i.value + n); }
  friend self operator + (difference_type n, self const& i) { return self(i.value + n); }

  // - operator
  friend difference_type operator - (self const& x, self const& y) { return x.value - y.value; }
  friend self operator - (self const& i, difference_type n) { return self(i.value - n); }

  // comparison operators
  friend bool operator == (self const& x, self const& y) { return x.value == y.value; }
  friend bool operator != (self const& x, self const& y) { return x.value != y.value; }
  friend bool operator <  (self const& x, self const& y) { return x.value <  y.value; }
  friend bool operator <= (self const& x, self const& y) { return x.value <= y.value; }
  friend bool operator >  (self const& x, self const& y) { return x.value >  y.value; }
  friend bool operator >= (self const& x, self const& y) { return x.value >= y.value; }

private:
  // THE iterator value
  value_type value;
};
