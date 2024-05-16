export const processedOneSpace = (str: string): string => {
  return str.replace(/[\\n\\r\\t]/g, ' ').replace(/\s+/g, ' ');
};

export const processedNoSpace = (str: string): string => {
  return str.replace(/\s+/g, '');
};
