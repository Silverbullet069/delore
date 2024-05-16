export interface ILoad<T> {
  load(path: string): Promise<void>;
}
