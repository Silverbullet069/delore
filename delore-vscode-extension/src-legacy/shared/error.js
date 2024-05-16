"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.makeRight = exports.makeLeft = exports.isRight = exports.isLeft = exports.unwrapEither = exports.isWeakNever = exports.isStrictNever = void 0;
const isStrictNever = (x) => {
    throw new Error(`
    Never case reached with unexpected value ${x}
  `);
};
exports.isStrictNever = isStrictNever;
const isWeakNever = (x) => {
    console.error(`Never case reached with unexpected value ${x} in ${new Error().stack}`);
};
exports.isWeakNever = isWeakNever;
const unwrapEither = (e) => {
    const { left, right } = e;
    if (right !== undefined && left !== undefined) {
        throw new Error(`
      Runtime Error! Received both left and right values at runtime when opening an Either\n
      Left: ${JSON.stringify(left)}\n
      Right: ${JSON.stringify(right)}\n
    `);
    }
    if (left !== undefined) {
        return left;
    }
    if (right !== undefined) {
        return right;
    }
    throw new Error(`
    Runtime Error! Received no left or right values at runtime when opening Either
  `);
};
exports.unwrapEither = unwrapEither;
const isLeft = (e) => {
    return e.left !== undefined;
};
exports.isLeft = isLeft;
const isRight = (e) => {
    return e.right !== undefined;
};
exports.isRight = isRight;
const makeLeft = (value) => ({ left: value });
exports.makeLeft = makeLeft;
const makeRight = (value) => ({ right: value });
exports.makeRight = makeRight;
