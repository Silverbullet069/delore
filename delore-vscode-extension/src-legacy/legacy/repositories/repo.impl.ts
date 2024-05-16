import { BaseRepository } from './base/repo.base';
import { Func } from '../model/func.model';

import { readFileSync } from 'fs';
import { writeFileSync } from 'fs';
import { writeFile as writeFileAsync } from 'fs/promises';
import { readFile as readFileAsync } from 'fs/promises';

import * as logger from '../../shared/logger';
import { basename } from 'path';

import { ID } from '../model/func.model';
import { AppState } from '../../../src/model/state.model';

export class FuncJsonRepository extends BaseRepository<Func> {
  // for now, there can be more than one FuncJsonRepository
  // e.g. connect into multiple JSON files

  // soft private
  private readonly _pathToJson: string;
  private readonly _funcsMap: Map<ID, Func>;

  public constructor(pathToJson: string) {
    super();
    this._pathToJson = pathToJson;
    this._funcsMap = new Map<ID, Func>();

    this.load();
  }

  public async load(): Promise<void> {
    try {
      const funcsJson: string = await readFileAsync(this._pathToJson, {
        encoding: 'utf8'
      });
      const funcs: Func[] = JSON.parse(funcsJson);
      funcs.forEach((func) => this._funcsMap.set(func.id, func));
    } catch (err) {
      logger.debugError(basename(module.filename), `load(): ${err}`);
    }
  }

  public async save(): Promise<void> {
    try {
      const funcs = Array.from(this._funcsMap.values());
      const funcsJson = JSON.stringify(funcs);

      await writeFileAsync(this._pathToJson, funcsJson);
    } catch (err) {
      logger.debugError(basename(module.filename), `save(): ${err}`);
    }
  }

  public async create(
    func: Func,
    isPersistent: boolean = false
  ): Promise<void> {
    try {
      // update cache
      const id = func.id;
      this._funcsMap.set(id, func);

      // update file
      if (isPersistent) {
        await this.save();
      }
    } catch (err) {
      logger.debugError(basename(module.filename), `create(): ${err}`);
    }
  }

  public async update(
    id: ID,
    func: Func,
    isPersistent: boolean = false
  ): Promise<void> {
    try {
      // update cache
      this._funcsMap.delete(id);
      this._funcsMap.set(id, func);

      // update file
      if (isPersistent) {
        await this.save();
      }
    } catch (err) {
      logger.debugError(basename(module.filename), `update(): ${err}`);
    }
  }

  public async delete(id: ID, isPersistent: boolean = false): Promise<void> {
    try {
      // update cache
      this._funcsMap.delete(id);

      // update file
      if (isPersistent) {
        await this.save();
      }
    } catch (err) {
      logger.debugError(basename(module.filename), `delete(): ${err}`);
    }
  }

  public async findAll(isRefreshFirst: boolean = false): Promise<Func[]> {
    if (isRefreshFirst) {
      await this.load();
    }

    return Array.from(this._funcsMap.values());
  }

  public async findById(
    id: ID,
    isRefreshFirst: boolean = false
  ): Promise<void | Func> {
    try {
      if (isRefreshFirst) {
        await this.load();
      }

      const func = this._funcsMap.get(id);
      if (func === undefined) {
        throw new Error(`ID not found! Check again!`);
      }

      return func;
    } catch (err) {
      logger.debugError(basename(module.filename), `findById: ${err}`);
    }
  }
}
