#ifndef _BUTTER_RNG_HPP_
#define _BUTTER_RNG_HPP_

#include <array>
#include <cstdint>
#include <initializer_list>
#include <istream>
#include <limits>
#include <ostream>
#include <random>
#include <type_traits>

namespace butter {
  namespace details {
    // Represents something that satisfies the named requirement SeedSequence.
    // Not exact, but enough for our purposes.
    template <class T>
    concept seed_sequence = std::is_unsigned_v<typename T::result_type> &&
      (std::numeric_limits<typename T::result_type>::digits >= 32) &&
      requires(T x, uint_least32_t* ib, uint_least32_t* ie) {
      x.generate(ib, ie);
    };
  }  // namespace details
  
  namespace tables {
    extern const std::array<uint16_t, 65114> rng_table;
  }
  
  class sm64_rng_engine {
  public:
    using result_type                         = uint16_t;
    static constexpr result_type default_seed = 0;

  private:
    result_type state;

  public:
    explicit sm64_rng_engine(result_type x) : state(x) {}
    template <details::seed_sequence T>
    sm64_rng_engine(T& x) :
      state([&]() {
        uint32_t seed;
        x.generate(&seed, (&seed + 1));
        return seed;
      }()) {}
    sm64_rng_engine(const sm64_rng_engine&) = default;
    sm64_rng_engine() : sm64_rng_engine(default_seed) {};

    void seed() { state = default_seed; }
    void seed(result_type x) { state = x; }
    void seed(const sm64_rng_engine& other) { state = other.state; }

    uint16_t operator()() {
      uint16_t t1, t2;

      if (state == 0x560A)
        state = 0;

      // no need to mask with 0x00FF/0xFF00, those bits are shifted out
      t1    = (state << 8) ^ state;
      state = (t1 << 8) | (t1 >> 8);

      t1 = ((t1 & 0x00FF) << 1) ^ state;
      t2 = (t1 >> 1) ^ 0xFF80;

      if ((t1 & 1) == 0)
        state = (t2 == 0xAA55) ? 0 : (t2 ^ 0x1FF4);
      else
        state = t2 ^ 0x8180;

      return state;
    }

    void discard(unsigned long long times) {
      for (unsigned long long i = 0; i < times; i++) {
        (*this)();
      }
    }

    result_type max() { return std::numeric_limits<result_type>::max(); }
    result_type min() { return std::numeric_limits<result_type>::min(); }

    friend bool operator==(
      const sm64_rng_engine& lhs, const sm64_rng_engine& rhs) {
      return lhs.state == rhs.state;
    }
    template <class CharT, class Traits>
    friend std::basic_ostream<CharT, Traits>& operator<<(
      std::basic_ostream<CharT, Traits>& out, const sm64_rng_engine& engine) {
      return (out << engine.state);
    }
    template <class CharT, class Traits>
    friend std::basic_istream<CharT, Traits>& operator<<(
      std::basic_istream<CharT, Traits>& in, const sm64_rng_engine& engine) {
      return (in >> engine.state);
    }

    uint16_t next_uint16() { return (*this)(); }
    float next_float() { return (*this)() / 65536.0; }
    uint32_t next_sign() { return ((*this)() > 32767) ? 1 : -1; }
  };
  
  
}  // namespace butter
#endif