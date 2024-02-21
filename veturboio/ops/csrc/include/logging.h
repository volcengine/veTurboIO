/*
 * Copyright (c) 2024 Beijing Volcano Engine Technology Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
using namespace std;

#define PR std::cout
#define ENDL std::endl
#define FILE_INFO "[" << __FUNCTION__ << " at " << __FILE__ << ":" << __LINE__ << "] "

#define ARG_COUNT_PRIVATE(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, N, ...) N
#define ARG_COUNT(...) ARG_COUNT_PRIVATE(0, __VA_ARGS__, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)

#define FUN_COUNT_GLUE(M, count) M##count
#define FUN_JOIN_COUNT(M, count) FUN_COUNT_GLUE(M, count)
#define FUN_JOIN_ARGS(x, y) x y
#define CallSomeOne(fn, ...) FUN_JOIN_ARGS(FUN_JOIN_COUNT(fn, ARG_COUNT(__VA_ARGS__)), (__VA_ARGS__))

#define param1(a) a
#define param2(a, b) a << ", " #b ":" << b
#define param3(a, b, c) a << ", " #b ":" << b << ", " #c ":" << c
#define param4(a, b, c, d) a << ", " #b ":" << b << ", " #c ":" << c << ", " #d ":" << d
#define param5(a, b, c, d, e) a << ", " #b ":" << b << ", " #c ":" << c << ", " #d ":" << d << ", " #e ":" << e

#define pr1(...) param1(__VA_ARGS__)
#define pr2(...) param2(__VA_ARGS__)
#define pr3(...) param3(__VA_ARGS__)
#define pr4(...) param4(__VA_ARGS__)
#define pr5(...) param5(__VA_ARGS__)

#define logDebug(...) PR << "VETURBOIO_CPP_DEBUG " << FILE_INFO << CallSomeOne(pr, __VA_ARGS__) << ENDL
#define logInfo(...) PR << "VETURBOIO_CPP_INFO " << FILE_INFO << CallSomeOne(pr, __VA_ARGS__) << ENDL
#define logWarn(...) PR << "VETURBOIO_CPP_WARN " << FILE_INFO << CallSomeOne(pr, __VA_ARGS__) << ENDL
#define logError(...) PR << "VETURBOIO_CPP_ERROR " << FILE_INFO << CallSomeOne(pr, __VA_ARGS__) << ENDL
#endif // LOGGER_H