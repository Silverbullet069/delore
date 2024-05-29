import * as logger from '../utils/logger';
import * as either from '../utils/either';

import {
  DetectionModelOutput,
  LineState,
  LocalizationModelOutput,
  RepairationModelLinesOutput,
  RepairationModelOutput
} from '../type/state.type';
import {
  Either,
  isLeft,
  makeLeft,
  makeRight,
  unwrapEither
} from '../utils/either';
import { areStringsSimilar, isOnlyWhitespace } from '../utils/misc';
import { processedNoSpace } from '../utils/sanitize';

export const mergeDetectionModelOutputs = (
  detectionResults: DetectionModelOutput[]
): DetectionModelOutput => {
  // Minimize False Negative is top-priority, we don't want to miss vuls
  // A func that get classified as vul by any model are considered vul
  // Of course, this can boost False Positive rate
  return detectionResults.reduce(
    (currentMergeDetectionResult, detectionResult) => {
      return {
        ...currentMergeDetectionResult,
        isVulnerable:
          currentMergeDetectionResult.isVulnerable ||
          detectionResult.isVulnerable
      };
    },
    {
      modelName: 'merge',
      isVulnerable: false
    }
  );
};

type MergeResultErrorType =
  | 'LINES_NOT_SAME_LENGTH'
  | 'LINE_NOT_SAME_CONTENT'
  | 'LINE_NOT_SAME_NUM'; // or LINES_NOT_SAME_NUM_ORDER

type MergeResultError = {
  type: MergeResultErrorType;
  msg: string;
};

const handleMergeLocalizationResult = (
  currentOutput: LocalizationModelOutput,
  nextOutput: LocalizationModelOutput
): Either<MergeResultError, LocalizationModelOutput> => {
  // currentOutput is reduce() initialized value
  if (
    currentOutput.modelName === 'merge' &&
    currentOutput.lines.length === 0 &&
    nextOutput.lines.length !== 0
  ) {
    return makeRight({ ...nextOutput, modelName: 'merge' });
  }

  // every model output must retain the number of lines and line order of function in editor
  if (currentOutput.lines.length !== nextOutput.lines.length) {
    return makeLeft({
      type: 'LINES_NOT_SAME_LENGTH',
      msg: `
        Check again both lines.
        Current output name: ${currentOutput.modelName}
        Current output lines length: ${currentOutput.lines.length}
        Next output name: ${nextOutput.modelName}
        Next output lines length: ${nextOutput.lines.length}
        ${new Error().stack}
      `
    });
  }

  const mergeLocalizationModelOutput: LocalizationModelOutput = {
    ...currentOutput, // current is chosen by design
    lines: []
  };

  // not very 'functional', but easy to read
  const length = currentOutput.lines.length;
  for (let i = 0; i < length; ++i) {
    // passed
    if (currentOutput.lines[i].content === nextOutput.lines[i].content) {
      continue;
    }

    // passed
    if (
      isOnlyWhitespace(currentOutput.lines[i].content) &&
      isOnlyWhitespace(nextOutput.lines[i].content)
    ) {
      continue;
    }

    if (
      currentOutput.lines[i].content !== nextOutput.lines[i].content &&
      !areStringsSimilar(
        processedNoSpace(currentOutput.lines[i].content),
        processedNoSpace(nextOutput.lines[i].content)
      )
    ) {
      return makeLeft({
        type: 'LINE_NOT_SAME_CONTENT',
        msg: `
          Check again both line.
          Current output name: ${currentOutput.modelName}
          Current output line content: ${currentOutput.lines[i].content}
          Next output name: ${nextOutput.modelName}
          Next output line content: ${nextOutput.lines[i].content}
          ${new Error().stack}
        `
      });
    }

    if (currentOutput.lines[i].num !== nextOutput.lines[i].num) {
      return makeLeft({
        type: 'LINE_NOT_SAME_NUM',
        msg: `
          Check again both line num.
          Current output name: ${currentOutput.modelName}
          Current output line num: ${currentOutput.lines[i].num}
          Next output name: ${nextOutput.modelName}
          Next output line num: ${nextOutput.lines[i].num}
          ${new Error().stack}
        `
      });
    }

    mergeLocalizationModelOutput.lines.push({
      num: currentOutput.lines[i].num, // current is chosen by design
      content: currentOutput.lines[i].content, // current is chosen by design
      score: 0, // score is meaningless when merge
      isVulnerable:
        currentOutput.lines[i].isVulnerable || nextOutput.lines[i].isVulnerable
    });
  }

  return makeRight(mergeLocalizationModelOutput);
};

export const mergeLocalizationModelOutputs = (
  localizationResults: LocalizationModelOutput[]
): LocalizationModelOutput => {
  return localizationResults.reduce(
    (currentMergeLocalizationResult, localizationResult) => {
      const mergeEither = handleMergeLocalizationResult(
        currentMergeLocalizationResult,
        localizationResult
      );

      if (isLeft(mergeEither)) {
        const err = unwrapEither(mergeEither);
        logger.debugError(err.type, '\n', err.msg);

        // skip the error model
        return currentMergeLocalizationResult;
      }

      const mergeOutput = unwrapEither(mergeEither);
      return mergeOutput;
    },
    {
      modelName: 'merge',
      lines: []
    }
  );
};

const handleMergeRepairationResult = (
  currentOutput: RepairationModelOutput,
  nextOutput: RepairationModelOutput
): Either<MergeResultError, RepairationModelOutput> => {
  // initialized value for
  if (
    currentOutput.modelName === 'merge' &&
    currentOutput.lines.length === 0 &&
    nextOutput.lines.length !== 0
  ) {
    nextOutput.lines = nextOutput.lines.map((line) => {
      return {
        ...line,
        cwe: '', // meaningless when merged
        reason: '', // meaningless when merged
        fix: !!line.isVulnerable
          ? ` // ${line.fix} (${line.cwe}: ${line.reason})`
          : ' //'
      } satisfies RepairationModelLinesOutput;
    });

    return makeRight({ ...nextOutput, modelName: 'merge' });
  }

  // every model output must retain the number of lines and line order of function in editor
  if (currentOutput.lines.length !== nextOutput.lines.length) {
    return makeLeft({
      type: 'LINES_NOT_SAME_LENGTH',
      msg: `
        Check again both lines.
        Current output name: ${currentOutput.modelName}
        Current output lines length: ${currentOutput.lines.length}
        Next output name: ${nextOutput.modelName}
        Next output lines length: ${nextOutput.lines.length}
        ${new Error().stack}
      `
    });
  }

  const mergeRepairationModelOutput: RepairationModelOutput = {
    ...currentOutput, // only the model name 'merge'
    lines: []
  };

  // not very 'functional', but easy to read
  const length = currentOutput.lines.length;
  for (let i = 0; i < length; ++i) {
    if (
      !areStringsSimilar(
        processedNoSpace(currentOutput.lines[i].content),
        processedNoSpace(nextOutput.lines[i].content)
      )
    ) {
      return makeLeft({
        type: 'LINE_NOT_SAME_CONTENT',
        msg: `
          Check again both line.
          Current output name: ${currentOutput.modelName}
          Current output line content: ${currentOutput.lines[i].content}
          Next output name: ${nextOutput.modelName}
          Next output line content: ${nextOutput.lines[i].content}
          ${new Error().stack}
        `
      });
    }

    if (currentOutput.lines[i].num !== nextOutput.lines[i].num) {
      return makeLeft({
        type: 'LINE_NOT_SAME_NUM',
        msg: `
          Check again both line num.
          Current output name: ${currentOutput.modelName}
          Current output line num: ${currentOutput.lines[i].num}
          Next output name: ${nextOutput.modelName}
          Next output line num: ${nextOutput.lines[i].num}
          ${new Error().stack}
        `
      });
    }

    // main logic
    mergeRepairationModelOutput.lines.push({
      num: nextOutput.lines[i].num, // current is chosen by design
      content: nextOutput.lines[i].content, // current is chosen by design
      cwe: '', // meaningless when merged
      reason: '', // meaningless when merged
      isVulnerable: false, // meaningless when merged
      fix: !!nextOutput.lines[i].isVulnerable
        ? `${currentOutput.lines[i].fix} // ${nextOutput.lines[i].fix} (${nextOutput.lines[i].cwe}: ${nextOutput.lines[i].reason})`
        : currentOutput.lines[i].fix
    } satisfies RepairationModelLinesOutput);
  }

  return makeRight(mergeRepairationModelOutput);
};

// for now, there isn't a method to merge RepairationModelOutput
export const mergeRepairationModelOutputs = (
  repairationResults: RepairationModelOutput[]
): RepairationModelOutput => {
  return repairationResults.reduce(
    (currentMergeRepairationResult, repairationResult) => {
      const mergeEither = handleMergeRepairationResult(
        currentMergeRepairationResult,
        repairationResult
      );

      if (isLeft(mergeEither)) {
        const err = unwrapEither(mergeEither);
        logger.debugError(err.type, '\n', err.msg);

        // skip the error model
        return currentMergeRepairationResult;
      }

      const mergeOutput = unwrapEither(mergeEither);
      return mergeOutput;
    },
    {
      modelName: 'merge',
      lines: []
    }
  );
};
