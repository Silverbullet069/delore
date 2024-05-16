/**
 * Cre: https://viblo.asia/p/generic-repository-trong-nodejs-voi-typescript-Do754M0QKM6
 *
 * Cre 2: https://softwareengineering.stackexchange.com/q/330824
 *
 * Also CQS-compatible
 */

export interface IRead<T> {
  findAll(): Promise<T[]>;
  findById(id: any): Promise<void | T>;
}
