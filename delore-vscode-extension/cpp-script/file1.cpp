int add(int a, int b) {
  return a + b;
}

char toUpperCase(char c) {
  if ('A' <= c && c <= 'Z') {
    return c;
  } else if ('a' <= c && c <= 'z')
    return c - ('a' - 'A');
  else
    throw('[ERROR] toUpperCase(char c): c not an alphabet!');
}