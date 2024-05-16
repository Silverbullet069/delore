// Cre: https://hackernoon.com/mastering-type-safe-json-serialization-in-typescript

import { Either, isStrictNever, makeLeft, makeRight } from './either';

/* ============================================ */
/* SERIALIZATION                                */
/* ============================================ */

type JSONPrimitive = string | number | boolean | null | undefined;
type JSONValue =
  | JSONPrimitive
  | JSONValue[]
  | {
      [key: string]: JSONValue;
    };

type JSONCompatible<T> = unknown extends T
  ? never
  : {
      [P in keyof T]: T[P] extends JSONValue
        ? T[P]
        : T[P] extends NotAssignableToJson
          ? never
          : JSONCompatible<T[P]>;
    };

type NotAssignableToJson = bigint | symbol | Function;

export const toJsonValue = <T>(value: JSONCompatible<T>): JSONValue => {
  return value;
};

export const safeJsonStringify = <T>(data: JSONCompatible<T>) => {
  return JSON.stringify(data);
};

/* ============================================ */
/* DESERIALIZATION                              */
/* ============================================ */

export const safeJsonParse = async (text: string) => {
  try {
    return (await JSON.parse(text)) as unknown;
  } catch (err) {
    return isStrictNever(err as never);
  }
};
