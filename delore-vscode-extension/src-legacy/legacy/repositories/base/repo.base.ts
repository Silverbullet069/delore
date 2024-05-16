import { ILoad } from '../interfaces/load.interface';
import { ISave } from '../interfaces/save.interface';
import { IRead } from '../interfaces/read.interface';
import { IWrite } from '../interfaces/write.interface';

export abstract class BaseRepository<T>
  implements IRead<T>, IWrite<T>, ILoad<T>, ISave<T>
{
  public async load(path: string): Promise<void> {
    throw new Error(`Method load() not implemented!`);
  }

  public async save(): Promise<void> {
    throw new Error(`Method save() not implemented!`);
  }

  public async create(item: T): Promise<void> {
    throw new Error(`Method create() not implemented!`);
  }

  public async update(id: any, item: T): Promise<void> {
    throw new Error(`Method update() not implemented!`);
  }

  public async delete(id: any): Promise<void> {
    throw new Error(`Method delete() not implemented!`);
  }

  public async findAll(): Promise<T[]> {
    throw new Error(`Method findAll() not implemented!`);
  }

  public async findById(id: any): Promise<T | void> {
    throw new Error(`Method findById() not implemented!`);
  }
}
