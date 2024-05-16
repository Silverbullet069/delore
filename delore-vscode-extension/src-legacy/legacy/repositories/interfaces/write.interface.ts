/**
 * Cre: https://viblo.asia/p/generic-repository-trong-nodejs-voi-typescript-Do754M0QKM6

 * Cre 2: https://softwareengineering.stackexchange.com/q/330824
 *
 * Also CQS-compatible
 */

export interface IWrite<T> {
  create(item: T): Promise<void>;
  update(id: any, item: T): Promise<void>;
  delete(id: any): Promise<void>;
}
