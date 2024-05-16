import * as logger from './logger';
import { basename } from 'path';

// Cre: https://antman-does-software.com/strict-and-weak-exhaustive-checks-in-typescript
export const isStrictNever = (x: never): never => {
  throw new Error(`
    Never case reached with unexpected value ${x}
  `);
};

export const isWeakNever = (x: never): void => {
  console.error(
    `Never case reached with unexpected value ${x} in ${new Error().stack}`
  );
};

// Cre: https://antman-does-software.com/stop-catching-errors-in-typescript-use-the-either-type-to-make-your-code-predictable

type Left<T> = {
  left: T;
  right?: never; // compile
};

type Right<U> = {
  left?: never; // compile
  right: U;
};

// Biased Either
export type Either<T, U> = NonNullable<Left<T> | Right<U>>; // compile + runtime

type UnwrapEither = <T, U>(e: Either<T, U>) => NonNullable<T | U>;

export const unwrapEither: UnwrapEither = <T, U>(e: Either<T, U>) => {
  const { left, right } = e;

  if (right !== undefined && left !== undefined) {
    logger.debugError(`${new Error().stack}`);

    throw new Error(`Runtime Error! Received both left and right values at runtime when opening an Either\n
      Left: ${JSON.stringify(left)}\n
      Right: ${JSON.stringify(right)}\n
    `);
  }

  if (left !== undefined) {
    return left as NonNullable<T>;
  }

  if (right !== undefined) {
    return right as NonNullable<U>;
  }

  logger.debugError(`${new Error().stack}`);
  throw new Error(`
    Runtime Error! Received no left or right values at runtime when opening Either
  `);

  // since
};

export const isLeft = <T, U>(e: Either<T, U>): e is Left<T> => {
  return e.left !== undefined;
};

export const isRight = <T, U>(e: Either<T, U>): e is Right<U> => {
  return e.right !== undefined;
};

export const makeLeft = <T>(value: T): Left<T> => ({ left: value });

export const makeRight = <U>(value: U): Right<U> => ({ right: value });
