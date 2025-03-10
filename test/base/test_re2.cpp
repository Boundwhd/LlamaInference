 
#include <re2/re2.h>
#include <iostream>
#include <assert.h>
 
int
main(void)
{
    int i;
    std::string s;
    assert(RE2::FullMatch("ruby:1234", "(\\w+):(\\d+)", &s, &i));
    assert(s == "ruby");
    assert(i == 1234);
 
    // Fails: "ruby" cannot be parsed as an integer.
    assert(!RE2::FullMatch("ruby", "(.+)", &i));
 
    // Success; does not extract the number.
    assert(RE2::FullMatch("ruby:1234", "(\\w+):(\\d+)", &s));
 
    // Success; skips NULL argument.
    assert(RE2::FullMatch("ruby:1234", "(\\w+):(\\d+)", (void*)NULL, &i));
 
    // Fails: integer overflow keeps value from being stored in i.
    assert(!RE2::FullMatch("ruby:123456789123", "(\\w+):(\\d+)", &s, &i));
 
    std::cout << "Ok" << std::endl;
    return 0;
}