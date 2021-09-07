import {
  IconButton,
  Tooltip,
  useColorModeValue as mode,
} from "@chakra-ui/react";
import { BiPointer } from "react-icons/bi";
import { useHotkeys } from "react-hotkeys-hook";

import { useLabellingStore, Tools } from "../../../connectors/labeling-state";

import { keymap } from "../../../keymap";

export type Props = {};

export const SelectionTool = () => {
  const isImageLoading = useLabellingStore((state) => state.isImageLoading);
  const selectedTool = useLabellingStore((state) => state.selectedTool);
  const setSelectedTool = useLabellingStore((state) => state.setSelectedTool);
  useHotkeys(
    keymap.toolSelect.key,
    () => setSelectedTool(Tools.SELECTION),
    {},
    []
  );

  return (
    <Tooltip
      label={`Selection tool [${keymap.toolSelect.key}]`}
      placement="right"
      openDelay={300}
    >
      <IconButton
        icon={<BiPointer size="1.3em" />}
        isDisabled={isImageLoading}
        role="checkbox"
        aria-checked={selectedTool === Tools.SELECTION}
        onClick={() => setSelectedTool(Tools.SELECTION)}
        backgroundColor={mode("white", "gray.800")}
        aria-label="Selection tool"
        pointerEvents="initial"
        isActive={selectedTool === Tools.SELECTION}
      />
    </Tooltip>
  );
};
