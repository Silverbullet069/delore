export type InstallationEnvironment = "Local" | "Remote";

export type LanguageSupport = "c" | "cpp";
export type ToggleLocalization = boolean;
export type ToggleRepairation = boolean;

export type DetectionModel = "Devign";
export type LocalizationModel = "LineVD";
export type RepairationModel = "VulRepair";

export type CustomDetectionModelBinaryAbsPaths = string[];
export type CustomLocalizationModelBinaryAbsPaths = string[];
export type CustomRepairationModelBinaryAbsPaths = string[];

export interface IConfig {
  detectionModel: DetectionModel;
  localizationModel: LocalizationModel;
  repairationModel: RepairationModel;
}

export interface IDetection {
  model: DetectionModel;
}

export interface ILocalization {
  model: LocalizationModel;
}

export interface IRepairation {
  model: RepairationModel;
}
