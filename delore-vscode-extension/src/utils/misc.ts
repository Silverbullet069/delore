export const capitalize = (str: string): string => {
  return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
};

/* ====================================================== */

const equalsIgnoreCase = (str1: string, str2: string): boolean => {
  return str1.toLowerCase() === str2.toLowerCase();
};

/* ====================================================== */

const levenshtein = (a: string, b: string) => {
  const matrix = [];

  let i;
  for (i = 0; i <= b.length; i++) {
    matrix[i] = [i];
  }

  let j;
  for (j = 0; j <= a.length; j++) {
    matrix[0][j] = j;
  }

  for (i = 1; i <= b.length; i++) {
    for (j = 1; j <= a.length; j++) {
      if (b.charAt(i - 1) === a.charAt(j - 1)) {
        matrix[i][j] = matrix[i - 1][j - 1];
      } else {
        matrix[i][j] = Math.min(
          matrix[i - 1][j - 1] + 1,
          Math.min(matrix[i][j - 1] + 1, matrix[i - 1][j] + 1)
        );
      }
    }
  }

  return matrix[b.length][a.length];
};

export const areStringsSimilar = (
  a: string,
  b: string,
  similarityThreshold = 0.7 // 70%
): boolean => {
  const distance = levenshtein(a, b);
  const longestLength = Math.max(a.length, b.length);
  const similarity = (longestLength - distance) / longestLength;
  return similarity >= similarityThreshold;
};

export const isOnlyWhitespace = (str: string): boolean => {
  return !/\S/.test(str);
};
