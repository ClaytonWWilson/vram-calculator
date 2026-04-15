export interface ModelInfo {
  _id: string;
  id: string;
  private: boolean;
  pipeline_tag: string;
  library_name: string;
  tags?: string[] | null;
  downloads: number;
  likes: number;
  modelId: string;
  author: string;
  sha: string;
  lastModified: string;
  gated: boolean;
  disabled: boolean;
  widgetData?: WidgetDataEntity[] | null;
  "model-index"?: null;
  config: Config;
  cardData: CardData;
  transformersInfo: TransformersInfo;
  gguf: Gguf;
  siblings?: SiblingsEntity[] | null;
  spaces?: string[] | null;
  createdAt: string;
  usedStorage: number;
}
export interface WidgetDataEntity {
  text: string;
}
export interface Config {}
export interface CardData {
  tags?: string[] | null;
  library_name: string;
  license: string;
  license_link: string;
  pipeline_tag: string;
  base_model?: string[] | null;
}
export interface TransformersInfo {
  auto_model: string;
}
export interface Gguf {
  total: number;
  architecture: string;
  context_length: number;
  chat_template: string;
  eos_token: string;
}
export interface SiblingsEntity {
  rfilename: string;
}
