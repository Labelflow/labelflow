import {
  Field,
  ID,
  InputType,
  IntersectionType,
  PartialType,
  PickType,
} from "@nestjs/graphql";
import { IsUUID } from "class-validator";
import { LabelClass } from "../../model/entities";

@InputType()
export class LabelClassCreateInput extends IntersectionType(
  PickType(LabelClass, ["name", "color"] as const, InputType),
  PartialType(PickType(LabelClass, ["id"] as const, InputType))
) {
  @Field(() => ID)
  @IsUUID()
  datasetId!: string;
}
