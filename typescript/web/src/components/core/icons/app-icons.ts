import { LabelType } from "@labelflow/graphql-types";
import { camelCase, pascalCase } from "change-case";
import {
  BiPencil,
  BiPointer,
  BiPurchaseTagAlt,
  BiShapePolygon,
  BiShapeSquare,
} from "react-icons/bi";
import { BsLightningFill, BsMegaphone } from "react-icons/bs";
import { HiSelector } from "react-icons/hi";
import { IoColorWandOutline, IoSearch } from "react-icons/io5";
import { IconType as ReactIconsIconType } from "react-icons/lib";
import { RiCloseFill, RiMoreLine } from "react-icons/ri";
import { CamelCase, PascalCase } from "type-fest";
import { Tools } from "../../../connectors/labeling-state";
import {
  IogAddSvg,
  IogDeleteSvg,
  IogInsideSvg,
  IogOutsideSvg,
  IogRemoveSvg,
  IogTargetSvg,
} from "./svg/index-svg";

export type IconType = ReactIconsIconType | SvgComponent;

export type EntityIcon<TEntity extends string, TProperty extends string> =
  `${CamelCase<TProperty>}${PascalCase<TEntity>}`;

export const getEntityIcon = <TEntity extends string, TProperty extends string>(
  entity: TEntity,
  label: TProperty
): EntityIcon<TEntity, TProperty> => {
  const propertyName = camelCase(label);
  const entityName = pascalCase(entity);
  return `${propertyName}${entityName}`;
};

export type LabelIconName = EntityIcon<"Label", LabelType>;

export const getLabelIconName = (label: LabelType): LabelIconName =>
  getEntityIcon("Label", label);

export const LABELS_ICONS: Record<LabelIconName, IconType> = {
  classificationLabel: BiPurchaseTagAlt,
  boxLabel: BiShapeSquare,
  polygonLabel: BiShapePolygon,
};

export type ToolIconName = EntityIcon<"Tool", Tools>;

export const getToolIconName = (tool: Tools): ToolIconName =>
  getEntityIcon("Tool", tool);

export const TOOLS_ICONS: Record<ToolIconName, IconType> = {
  selectTool: BiPointer,
  classificationTool: LABELS_ICONS.classificationLabel,
  boxTool: LABELS_ICONS.boxLabel,
  freehandTool: BiPencil,
  polygonTool: LABELS_ICONS.polygonLabel,
  iogTool: IoColorWandOutline,
  aiAssistantTool: BsLightningFill,
};

export type IogIconName = `iog${PascalCase<
  "add" | "delete" | "inside" | "outside" | "remove" | "target"
>}`;

const IOG_ICONS: Record<IogIconName, IconType> = {
  iogAdd: IogAddSvg,
  iogDelete: IogDeleteSvg,
  iogInside: IogInsideSvg,
  iogOutside: IogOutsideSvg,
  iogRemove: IogRemoveSvg,
  iogTarget: IogTargetSvg,
};

export type AppIcon =
  | "announcements"
  | "close"
  | "more"
  | "search"
  | "selector"
  | ToolIconName
  | LabelIconName
  | IogIconName;

export const APP_ICONS: Record<AppIcon, IconType> = {
  ...TOOLS_ICONS,
  ...LABELS_ICONS,
  ...IOG_ICONS,
  announcements: BsMegaphone,
  close: RiCloseFill,
  more: RiMoreLine,
  search: IoSearch,
  selector: HiSelector,
};
