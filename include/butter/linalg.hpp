#ifndef _BUTTER_LINALG_HPP_
#define _BUTTER_LINALG_HPP_

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include <butter/support.hpp>

namespace butter {
  namespace details {

    template <class T, size_t L, class ISeq = std::make_index_sequence<L>>
    class vec_impl;

    template <class T, size_t L, size_t... Is>
    class vec_impl<T, L, std::index_sequence<Is...>> {
      template <
        class T_, size_t R, size_t C, class RSeq, class CSeq, class ESeq>
      friend class mat_impl;

    public:
      using storage_type = std::array<T, L>;
      using value_type   = T;

    protected:
      storage_type m_data;

      BUTTER_FORCE_INLINE vec_impl(const storage_type& data) : m_data(data) {};
      BUTTER_FORCE_INLINE vec_impl(storage_type&& data) :
        m_data(std::move(data)) {};

    public:
      // Proxy reference, for consistency with SIMDized versions.
      class reference {
        T& ref;

      public:
        operator T() { return ref; }

        template <class U>
        T operator=(U&& value) {
          ref = std::forward<U>(value);
        }
      };

      BUTTER_FORCE_INLINE vec_impl(sink_index<T, Is>... values) :
        m_data {values...} {}

      BUTTER_FORCE_INLINE vec_impl() : vec_impl((void(Is), T())...) {}

      BUTTER_FORCE_INLINE reference operator[](size_t x) {
        return reference {m_data[x]};
      }
      BUTTER_FORCE_INLINE reference x() requires(L >= 1) {
        return reference {m_data[0]};
      }
      BUTTER_FORCE_INLINE reference y() requires(L >= 2) {
        return reference {m_data[1]};
      }
      BUTTER_FORCE_INLINE reference z() requires(L >= 3) {
        return reference {m_data[2]};
      }
      BUTTER_FORCE_INLINE reference w() requires(L >= 4) {
        return reference {m_data[3]};
      }

    protected :

      BUTTER_FORCE_INLINE static storage_type
      add(storage_type a, storage_type b) {
        return storage_type {(a[Is] + b[Is])...};
      }
      BUTTER_FORCE_INLINE static storage_type sub(
        storage_type a, storage_type b) {
        return storage_type {(a[Is] - b[Is])...};
      }
      BUTTER_FORCE_INLINE static value_type mul(storage_type a, value_type b) {
        return storage_type {(a[Is] * b)...};
      }
      BUTTER_FORCE_INLINE static value_type div(storage_type a, value_type b) {
        return storage_type {(a[Is] / b)...};
      }
      BUTTER_FORCE_INLINE static value_type dot(
        storage_type a, storage_type b) {
        return (... + (a[Is] * b[Is]));
      }
      BUTTER_FORCE_INLINE static storage_type cross(
        storage_type a, storage_type b) requires(L == 3) {
        return storage_type {
          a[1] * b[2] - b[1] * a[2],
          a[0] * b[2] - b[0] * a[2],
          a[0] * b[1] - b[0] * a[1],
        };
      }
      BUTTER_FORCE_INLINE static storage_type norm(storage_type a) requires(
        std::is_floating_point_v<T>) {
        value_type dist = std::sqrt(dot(a, a));
        return storage_type {(a[Is] / dist)...};
      }
    };
  }  // namespace details

  template <class T, size_t L>
    requires(std::is_arithmetic_v<T> && 2 <= L && L <= 4)
  class vec : private details::vec_impl<T, L> {
    template <class T_, size_t R, size_t C>
      requires(std::is_arithmetic_v<T_>&& R >= 2 && R <= 4 && C >= 2 && C <= 4)
    friend class mat;
  private:
    using details::vec_impl<T, L>::m_data;

  public:
    using typename details::vec_impl<T, L>::value_type;

    using details::vec_impl<T, L>::vec_impl;
    using typename details::vec_impl<T, L>::reference;

    using details::vec_impl<T, L>::operator[];
    using details::vec_impl<T, L>::x;
    using details::vec_impl<T, L>::y;
    using details::vec_impl<T, L>::z;
    using details::vec_impl<T, L>::w;

    BUTTER_FORCE_INLINE friend vec operator+(vec a, vec b) {
      return vec(details::vec_impl<T, L>::add(a.m_data, b.m_data));
    }
    BUTTER_FORCE_INLINE friend vec& operator+=(vec& a, vec b) {
      a.m_data = details::vec_impl<T, L>::add(a.m_data, b.m_data);
      return a;
    }

    BUTTER_FORCE_INLINE friend vec operator-(vec a, vec b) {
      return vec(details::vec_impl<T, L>::sub(a.m_data, b.m_data));
    }
    BUTTER_FORCE_INLINE friend vec& operator-=(vec& a, vec b) {
      a.m_data = details::vec_impl<T, L>::sub(a.m_data, b.m_data);
      return a;
    }

    BUTTER_FORCE_INLINE friend vec operator*(vec a, value_type b) {
      return vec(details::vec_impl<T, L>::mul(a.m_data, b));
    }
    BUTTER_FORCE_INLINE friend vec operator*(value_type a, vec b) {
      return vec(details::vec_impl<T, L>::mul(b.m_data, a));
    }
    BUTTER_FORCE_INLINE friend vec& operator*=(vec& a, value_type b) {
      a.m_data = details::vec_impl<T, L>::mul(a.m_data, b);
      return a;
    }

    BUTTER_FORCE_INLINE friend vec operator/(vec a, value_type b) {
      return vec(details::vec_impl<T, L>::div(a.m_data, b));
    }
    BUTTER_FORCE_INLINE friend vec& operator/=(vec& a, value_type b) {
      a.m_data = details::vec_impl<T, L>::div(a.m_data, b);
      return a;
    }

    friend value_type dot(vec a, vec b);
    friend vec cross(vec a, vec b) requires(L == 3);
    friend vec norm(vec a) requires(std::is_floating_point_v<T>);
  };

  template <class T, size_t L>
  typename vec<T, L>::value_type dot(vec<T, L> a, vec<T, L> b) {
    return vec<T, L>(details::vec_impl<T, L>::dot(a.m_data, b.m_data));
  }
  template <class T, size_t L>
  vec<T, L> cross(vec<T, L> a, vec<T, L> b) requires(L == 3) {
    return vec<T, L>(details::vec_impl<T, L>::cross(a.m_data, b.m_data));
  }
  template <class T, size_t L>
  vec<T, L> norm(vec<T, L> a) requires(std::is_floating_point_v<T>) {
    return vec<T, L>(details::vec_impl<T, L>::norm(a.m_data));
  }

  struct index_pair {
    size_t col, row;
  };

  namespace details {
    template <
      class T, size_t R, size_t C, class RSeq = std::make_index_sequence<R>,
      class CSeq = std::make_index_sequence<C>,
      class ESeq = std::make_index_sequence<R * C>>
    class mat_impl;

    template <
      class T, size_t R, size_t C, size_t... Rs, size_t... Cs, size_t... Es>
    class mat_impl<
      T, R, C, std::index_sequence<Rs...>, std::index_sequence<Cs...>,
      std::index_sequence<Es...>> {
    public:
      using storage_type = std::array<T, R * C>;
      using value_type   = T;
      using column_type  = vec<T, C>;

    protected:
      storage_type m_data;
      
      BUTTER_FORCE_INLINE mat_impl(const storage_type& data) : m_data(data) {};
      BUTTER_FORCE_INLINE mat_impl(storage_type&& data) :
        m_data(std::move(data)) {};
    public:
      // Proxy reference, for consistency with SIMDized versions.
      class reference {
        T& ref;

      public:
        operator T() { return ref; }

        template <class U>
        T operator=(U&& value) {
          ref = std::forward<U>(value);
        }
      };
      class col_reference {
        T* const ref;

      public:
        reference operator[](size_t i) { return reference {ref[i]}; }
        friend col_reference& operator+=(col_reference& a, column_type b) {
          (void(a.ref[Rs] += b[Rs]), ...);
          return a;
        }
        friend col_reference& operator-=(col_reference& a, column_type b) {
          (void(a.ref[Rs] -= b[Rs]), ...);
          return a;
        }
        friend col_reference& operator*=(col_reference& a, value_type b) {
          (void(a.ref[Rs] *= b), ...);
          return a;
        }
        friend col_reference& operator/=(col_reference& a, value_type b) {
          (void(a.ref[Rs] /= b), ...);
          return a;
        }

        operator column_type() { return column_type {ref[Rs]...}; }

        col_reference& operator=(column_type rhs) {
          (void(ref[Rs] = rhs[Rs]), ...);
          return *this;
        }
      };

      mat_impl(sink_index<T, Es>... vs) : m_data {vs...} {}
      mat_impl() : m_data {(void(Es), T())...} {}

      reference operator[](index_pair indices) {
        return reference {m_data[indices.col * R + indices.row]};
      }
      reference operator[](size_t index) {
        return col_reference {&m_data[index * R]};
      }

    protected:
      static storage_type add(const storage_type& x, const storage_type& y) {
        return storage_type {(x[Es] + y[Es])...};
      }
      static void addeq(storage_type& x, const storage_type& y) {
        (void(x[Es] += y[Es]), ...);
      }
      static storage_type sub(const storage_type& x, const storage_type& y) {
        return storage_type {(x[Es] - y[Es])...};
      }
      static void subeq(storage_type& x, const storage_type& y) {
        (void(x[Es] -= y[Es]), ...);
      }
      static storage_type mul(const storage_type& x, value_type y) {
        return storage_type {(x[Es] * y)...};
      }
      static storage_type muleq(storage_type& x, value_type y) {
        (void(x[Es] *= y), ...);
      }
      static storage_type div(const storage_type& x, value_type y) {
        return storage_type {(x[Es] / y)...};
      }
      static storage_type diveq(storage_type& x, value_type y) {
        (void(x[Es] /= y), ...);
      }

      static typename vec_impl<T, R>::storage_type vmul(
        const storage_type& x, typename vec_impl<T, R>::storage_type y) {
        // clang-format off
        // Need lambda to isolate expansion of Rs from Cs
        return typename vec_impl<T, R>::storage_type {
          [&](size_t off) { return ((x[Rs * R + off] * y[Rs]) + ...); }(Rs)...
        };
        // clang-format on
      }

      template <size_t Q>
      static typename mat_impl<T, Q, R>::storage_type matmul(
        const storage_type& x,
        const typename mat_impl<T, Q, R>::storage_type& y) {
        // clang-format off
        // Outer lambda adds a parameter pack for indices of the destination array.
        // Inner lambda takes row and column index and calculates that element of the matrix product.
        // Inner lambda is needed to prevent zipping pack expansion.
        return [&]<size_t... Ds>(std::index_sequence<Ds...>) {
          return typename mat_impl<T, Q, C>::storage_type {
            [&]<size_t OR, size_t OC>(std::index_sequence<OR, OC>) {
              return ((x[Rs * R + OR] * y[OC * Q + Rs]) + ...);
            } (std::index_sequence<Ds / C, Ds % C> {})...
          };
        } (std::make_index_sequence<C * Q> {});
        // clang-format on
      }
      
      
    };
  }  // namespace details

  template <class T, size_t R, size_t C>
    requires(std::is_arithmetic_v<T>&& R >= 2 && R <= 4 && C >= 2 && C <= 4)
  class mat : private details::mat_impl<T, R, C> {
  private:
    using details::mat_impl<T, R, C>::m_data;
  public:
    using typename details::mat_impl<T, R, C>::value_type;
  
    using details::mat_impl<T, R, C>::mat_impl;
    using typename details::mat_impl<T, R, C>::reference;
    using typename details::mat_impl<T, R, C>::col_reference;
    
    using details::mat_impl<T, R, C>::operator[];
    
    friend mat operator+(const mat& a, const mat& b) {
      return mat(details::mat_impl<T, R, C>::add(a.m_data, b.m_data));
    }
    friend mat& operator+=(mat& a, const mat& b) {
      details::mat_impl<T, R, C>::addeq(a.m_data, b.m_data);
      return a;
    }
    
    friend mat operator-(const mat& a, const mat& b) {
      return mat(details::mat_impl<T, R, C>::sub(a.m_data, b.m_data));
    }
    friend mat& operator-=(mat& a, const mat& b) {
      details::mat_impl<T, R, C>::subeq(a.m_data, b.m_data);
      return a;
    }
    
    friend mat operator*(const mat& a, value_type b) {
      return mat(details::mat_impl<T, R, C>::mul(a.m_data, b));
    }
    friend mat operator*(value_type a, const mat& b) {
      return mat(details::mat_impl<T, R, C>::mul(b.m_data, a));
    }
    friend mat& operator*=(mat& a, value_type b) {
      details::mat_impl<T, R, C>::muleq(a.m_data, b);
      return a;
    }
    
    friend mat operator/(const mat& a, value_type b) {
      return mat(details::mat_impl<T, R, C>::div(a.m_data, b));
    }
    friend mat& operator/=(mat& a, value_type b) {
      details::mat_impl<T, R, C>::diveq(a.m_data, b);
      return a;
    }
    
    friend vec<T, R> operator*(const mat& a, const vec<T, C>& b) {
      return vec<T, R>(details::mat_impl<T, R, C>::vmul(a.m_data, b.m_data));
    }
    
    template <size_t Q>
    friend mat<T, Q, C> operator*(const mat& a, const mat<T, Q, R>& b) {
      return mat<T, Q, C>(details::mat_impl<T, R, C>::template matmul<Q>(a.m_data, b.m_data));
    }
  };
}  // namespace butter
#endif