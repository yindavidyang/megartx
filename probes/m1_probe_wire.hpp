// Address-free host introspection helpers. No CUDA headers, calls or allocations.
#pragma once
#include <cstdint>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace megartx_probe {
inline void name(std::ostream& out, std::string const& value) {
  if (value.empty() || value.size() > 96) throw std::invalid_argument("invalid field name");
  for (char c : value)
    if (!(c == '_' || c == '.' || (c >= 'a' && c <= 'z') ||
          (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')))
      throw std::invalid_argument("invalid field name");
  out << '"' << value << '"';
}
template <class T> void type(std::ostream& out) {
  out << "{\"size\":" << sizeof(T) << ",\"align\":" << alignof(T) << '}';
}
template <class Object, class Member>
void member(std::ostream& out, Object const& object, Member const& value) {
  auto begin = reinterpret_cast<std::uintptr_t>(&object);
  auto address = reinterpret_cast<std::uintptr_t>(&value);
  if (address < begin || address - begin > sizeof(Object) ||
      sizeof(Member) > sizeof(Object) - (address - begin))
    throw std::invalid_argument("member outside its object");
  // Measure the typed object's representation; do not assume standard-layout offsetof.
  out << "{\"offset\":" << address - begin << ",\"size\":" << sizeof(Member)
      << ",\"align\":" << alignof(Member) << '}';
}
inline bool little_endian() {
  std::uint16_t word = 1;
  return *reinterpret_cast<unsigned char const*>(&word) == 1;
}
}  // namespace megartx_probe
