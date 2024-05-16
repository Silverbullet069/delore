"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.BaseRepository = void 0;
class BaseRepository {
    async load(path) {
        throw new Error(`Method load() not implemented!`);
    }
    async save() {
        throw new Error(`Method save() not implemented!`);
    }
    async create(item) {
        throw new Error(`Method create() not implemented!`);
    }
    async update(id, item) {
        throw new Error(`Method update() not implemented!`);
    }
    async delete(id) {
        throw new Error(`Method delete() not implemented!`);
    }
    async findAll() {
        throw new Error(`Method findAll() not implemented!`);
    }
    async findById(id) {
        throw new Error(`Method findById() not implemented!`);
    }
}
exports.BaseRepository = BaseRepository;
