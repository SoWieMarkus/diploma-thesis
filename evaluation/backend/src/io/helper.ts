import * as fs from "fs";
import path from "path";

export interface IVector {
    x: number;
    y: number;
    z: number;
}

export interface Directory {
    path: string;
    name: string;
}

export const OUTPUT_PATH = path.join(__dirname, '../../../../../output');

export const getVectorFromString = (line: string) => {
    const numbers = line.split(",").map(num => parseFloat(num));
    if (numbers.length !== 3) throw new Error("Illegal State: Array is too short.");
    return {
        x: numbers[0],
        y: numbers[1],
        z: numbers[2]
    }
}

export const getSubDirectories = (directory: string) => {
    const subDirectories = fs.readdirSync(directory)
        .map(file => path.join(directory, file))
        .filter(path => fs.statSync(path).isDirectory());
    const outputDirectories = []
    for (const subDirectory of subDirectories) {
        outputDirectories.push({
            path: subDirectory,
            name: path.basename(subDirectory)
        })
    }
    return outputDirectories;
}

export const getFilesWithExtension = (dir: string, extension: string): Directory[] => {
    const files: Directory[] = [];
    const entries = fs.readdirSync(dir);

    for (const entry of entries) {
        const entryPath = path.join(dir, entry);
        const stat = fs.statSync(entryPath);

        if (!stat.isDirectory()) {
            files.push({
                path: entryPath,
                name: entry
            })
        }
    }

    return files;
}








