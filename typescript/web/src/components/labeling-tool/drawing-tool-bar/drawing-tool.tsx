import {
  useState,
  useEffect,
  useRef,
  useCallback,
  Ref,
  MouseEventHandler,
} from "react";
import {
  Tooltip,
  Popover,
  PopoverContent,
  Button,
  PopoverBody,
  PopoverTrigger,
  useColorModeValue,
  Box,
  Kbd,
  Flex,
  Text,
  chakra,
} from "@chakra-ui/react";
import { useCookies } from "react-cookie";
import { RiArrowDownSLine } from "react-icons/ri";
import {
  BiShapeSquare,
  BiPurchaseTagAlt,
  BiShapePolygon,
  BiPencil,
} from "react-icons/bi";
import { IoColorWandOutline } from "react-icons/io5";
import { useHotkeys } from "react-hotkeys-hook";

import { useLabelingStore, Tools } from "../../../connectors/labeling-state";
import { IogAlertDialog } from "./iog-alert-dialog";

import { keymap } from "../../../keymap";

export type Props = {};

const ChakraBiLabel = chakra(BiPurchaseTagAlt);
const ChakraBiShapeSquare = chakra(BiShapeSquare);
const ChakraBiShapePolygon = chakra(BiShapePolygon);
const ChakraRiArrowDownSLine = chakra(RiArrowDownSLine);
const ChakraIoColorWandOutline = chakra(IoColorWandOutline);
const ChakraBiPencil = chakra(BiPencil);

export type ToolSelectionPopoverItemProps = {
  name: string;
  shortcut: string;
  selected?: boolean;
  onClick: any;
  children: any;
  ariaLabel: string;
};

type UseItemColorsOptions = Pick<ToolSelectionPopoverItemProps, "selected">;

const useBackgroundColors = ({
  selected,
}: UseItemColorsOptions): [string, string] =>
  selected ? ["gray.300", "gray.500"] : ["gray.100", "gray.600"];

const useBackgroundColor = (options: UseItemColorsOptions): string =>
  useColorModeValue(...useBackgroundColors(options));

const useHoverBackgroundColors = ({
  selected,
}: UseItemColorsOptions): [string, string] =>
  selected ? ["gray.300", "gray.500"] : ["transparent", "transparent"];

const useHoverBackgroundColor = (options: UseItemColorsOptions): string =>
  useColorModeValue(...useHoverBackgroundColors(options));

type UseItemColorsResult = Record<"bgColor" | "hoverBgColor", string>;

const useItemColors = (options: UseItemColorsOptions): UseItemColorsResult => ({
  bgColor: useBackgroundColor(options),
  hoverBgColor: useHoverBackgroundColor(options),
});

export const ToolSelectionPopoverItem = ({
  name,
  shortcut,
  selected,
  children,
  onClick,
  ariaLabel,
}: ToolSelectionPopoverItemProps) => {
  const { bgColor, hoverBgColor } = useItemColors({ selected });
  return (
    <Box
      pl="0"
      pr="0"
      pt="1"
      pb="1"
      role="checkbox"
      aria-label={ariaLabel}
      aria-checked={selected}
      _hover={{ backgroundColor: hoverBgColor }}
      bgColor={bgColor}
      onClick={() => {
        onClick();
      }}
    >
      <Flex justifyContent="space-between" alignItems="center">
        {children}
        <Text flexGrow={1} whiteSpace="nowrap" ml="2" mr="2">
          {name}
        </Text>

        <Kbd flexShrink={0} flexGrow={0} justifyContent="center" mr="2">
          {shortcut}
        </Kbd>
      </Flex>
    </Box>
  );
};

export const DrawingToolIcon = (props: {
  isDisabled: boolean;
  selectedTool: Tools;
  setSelectedTool: any;
  onClickDetails: MouseEventHandler<HTMLButtonElement>;
  buttonRef: Ref<HTMLButtonElement>;
}) => {
  const {
    isDisabled,
    onClickDetails,
    selectedTool,
    setSelectedTool,
    buttonRef,
  } = props;
  const [lastTool, setLastTool] = useState(Tools.BOX);
  useEffect(() => {
    if (
      [
        Tools.CLASSIFICATION,
        Tools.BOX,
        Tools.POLYGON,
        Tools.IOG,
        Tools.FREEHAND,
      ].includes(selectedTool)
    ) {
      setLastTool(selectedTool);
    }
  }, [selectedTool]);
  const isActive = [
    Tools.CLASSIFICATION,
    Tools.BOX,
    Tools.POLYGON,
    Tools.IOG,
    Tools.FREEHAND,
  ].includes(selectedTool);

  let toolTipLabel;
  switch (lastTool) {
    case Tools.CLASSIFICATION:
      toolTipLabel = `Classification tool [${keymap.toolClassification.key}]`;
      break;
    case Tools.BOX:
      toolTipLabel = `Bounding Box tool [${keymap.toolBoundingBox.key}]`;
      break;
    case Tools.POLYGON:
      toolTipLabel = `Polygon tool [${keymap.toolPolygon.key}]`;
      break;
    case Tools.IOG:
      toolTipLabel = `Auto Polygon tool [${keymap.toolIog.key}]`;
      break;
    case Tools.FREEHAND:
      toolTipLabel = `Freehand tool [${keymap.toolFreehand.key}]`;
      break;
    default:
      toolTipLabel = `Bounding Box tool [${keymap.toolBoundingBox.key}]`;
      break;
  }

  return (
    <Tooltip label={toolTipLabel} placement="right" openDelay={300}>
      <Button
        ref={buttonRef}
        isDisabled={isDisabled}
        role="checkbox"
        aria-checked={[
          Tools.CLASSIFICATION,
          Tools.BOX,
          Tools.POLYGON,
          Tools.IOG,
          Tools.FREEHAND,
        ].includes(selectedTool)}
        backgroundColor={useColorModeValue("white", "gray.800")}
        aria-label={`Drawing ${lastTool} tool`}
        pointerEvents="initial"
        onClick={() => setSelectedTool(lastTool)}
        isActive={isActive}
        w="10"
        padding="0"
      >
        {(() => {
          switch (lastTool) {
            case Tools.CLASSIFICATION:
              return <ChakraBiLabel size="1.3em" />;
            case Tools.BOX:
              return <ChakraBiShapeSquare size="1.3em" />;
            case Tools.POLYGON:
              return <ChakraBiShapePolygon size="1.3em" />;
            case Tools.IOG:
              return <ChakraIoColorWandOutline size="1.3em" />;
            case Tools.FREEHAND:
              return <ChakraBiPencil size="1.3em" />;
            default:
              return <ChakraBiShapeSquare size="1.3em" />;
          }
        })()}

        <PopoverTrigger>
          <Button
            as="div"
            variant="link"
            position="absolute"
            right="0"
            bottom="0"
            height="4"
            width="4"
            minWidth="4"
            maxWidth="4"
            isDisabled={isDisabled}
            role="button"
            aria-label="Change Drawing tool"
            pointerEvents="initial"
            onClick={onClickDetails}
            isActive={isActive}
            padding="0"
            textAlign="right"
            color={useColorModeValue("gray.800", "gray.200")}
          >
            <ChakraRiArrowDownSLine
              position="relative"
              size="0.7em"
              transform="translate( 0.01em, 0.01em) rotate(-45deg) scale(2) "
            />
          </Button>
        </PopoverTrigger>
      </Button>
    </Tooltip>
  );
};

export const DrawingTool = () => {
  const isImageLoading = useLabelingStore((state) => state.isImageLoading);
  const selectedTool = useLabelingStore((state) => state.selectedTool);
  const setSelectedTool = useLabelingStore((state) => state.setSelectedTool);
  const setSelectedLabelId = useLabelingStore(
    (state) => state.setSelectedLabelId
  );
  const [isPopoverOpened, doSetIsPopoverOpened] = useState(false);
  const [isIogAlertDialogOpen, doSetIsIogAlertDialogOpen] = useState(false);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const setIsPopoverOpened = useCallback((newState) => {
    doSetIsPopoverOpened(newState);
    if (!newState && buttonRef.current) {
      buttonRef.current.focus();
    }
  }, []);
  const [{ hasUserAcceptedIog }, setHasUserAcceptedIog] = useCookies([
    "hasUserAcceptedIog",
  ]);

  useEffect(() => {
    if (selectedTool !== Tools.SELECTION) {
      setSelectedLabelId(null);
    }
    if (selectedTool === Tools.IOG) {
      if (hasUserAcceptedIog !== "true") {
        doSetIsIogAlertDialogOpen(true);
      }
    }
  }, [selectedTool]);

  useHotkeys(
    keymap.toolClassification.key,
    () => {
      setSelectedTool(Tools.CLASSIFICATION);
    },
    {},
    [setSelectedTool]
  );
  useHotkeys(
    keymap.toolBoundingBox.key,
    () => {
      setSelectedTool(Tools.BOX);
    },
    {},
    [setSelectedTool]
  );
  useHotkeys(
    keymap.toolPolygon.key,
    () => {
      setSelectedTool(Tools.POLYGON);
    },
    {},
    [setSelectedTool]
  );
  useHotkeys(
    keymap.toolFreehand.key,
    () => {
      setSelectedTool(Tools.FREEHAND);
    },
    {},
    [setSelectedTool]
  );
  useHotkeys(
    keymap.toolIog.key,
    () => {
      setSelectedTool(Tools.IOG);
    },
    {},
    [setSelectedTool]
  );
  return (
    <>
      <IogAlertDialog
        isOpen={isIogAlertDialogOpen}
        onClose={() => {
          doSetIsIogAlertDialogOpen(false);
        }}
        onAccept={() => {
          setHasUserAcceptedIog("hasUserAcceptedIog", "true", {
            path: "/",
            httpOnly: false,
          });
        }}
        onCancel={() => {
          setSelectedTool(Tools.SELECTION);
        }}
      />
      <Popover
        isOpen={isPopoverOpened}
        placement="right-start"
        closeOnBlur
        onClose={() => {
          setIsPopoverOpened(false);
        }}
      >
        <DrawingToolIcon
          buttonRef={buttonRef}
          isDisabled={isImageLoading}
          onClickDetails={(e) => {
            setIsPopoverOpened(!isPopoverOpened);
            e.stopPropagation();
          }}
          selectedTool={selectedTool}
          setSelectedTool={setSelectedTool}
        />
        <PopoverContent
          borderColor={useColorModeValue("gray.200", "gray.600")}
          cursor="default"
          pointerEvents="initial"
          aria-label="Change Drawing Tool"
          width="60"
        >
          <PopoverBody pl="0" pr="0">
            <Box>
              <ToolSelectionPopoverItem
                name="Classification"
                shortcut={keymap.toolClassification.key}
                selected={selectedTool === Tools.CLASSIFICATION}
                onClick={() => {
                  setSelectedTool(Tools.CLASSIFICATION);
                  setIsPopoverOpened(false);
                }}
                ariaLabel="Classification tool"
              >
                <Box ml="2">
                  <ChakraBiLabel size="1.3em" />
                </Box>
              </ToolSelectionPopoverItem>
              <ToolSelectionPopoverItem
                name="Bounding Box"
                shortcut={keymap.toolBoundingBox.key}
                selected={selectedTool === Tools.BOX}
                onClick={() => {
                  setSelectedTool(Tools.BOX);
                  setIsPopoverOpened(false);
                }}
                ariaLabel="Bounding box tool"
              >
                <Box ml="2">
                  <ChakraBiShapeSquare size="1.3em" />
                </Box>
              </ToolSelectionPopoverItem>
              <ToolSelectionPopoverItem
                name="Polygon"
                shortcut={keymap.toolPolygon.key}
                selected={selectedTool === Tools.POLYGON}
                onClick={() => {
                  setSelectedTool(Tools.POLYGON);
                  setIsPopoverOpened(false);
                }}
                ariaLabel="Polygon tool"
              >
                <Box ml="2">
                  <ChakraBiShapePolygon size="1.3em" />
                </Box>
              </ToolSelectionPopoverItem>
              <ToolSelectionPopoverItem
                name="Freehand"
                shortcut={keymap.toolFreehand.key}
                selected={selectedTool === Tools.FREEHAND}
                onClick={() => {
                  setSelectedTool(Tools.FREEHAND);
                  setIsPopoverOpened(false);
                }}
                ariaLabel="Freehand tool"
              >
                <Box ml="2">
                  <ChakraBiPencil size="1.3em" />
                </Box>
              </ToolSelectionPopoverItem>
              <ToolSelectionPopoverItem
                name="Auto Polygon"
                shortcut={keymap.toolIog.key}
                selected={selectedTool === Tools.IOG}
                onClick={() => {
                  setSelectedTool(Tools.IOG);
                  setIsPopoverOpened(false);
                }}
                ariaLabel="Auto polygon Tool"
              >
                <Box ml="2">
                  <ChakraIoColorWandOutline size="1.3em" />
                </Box>
              </ToolSelectionPopoverItem>
            </Box>
          </PopoverBody>
        </PopoverContent>
      </Popover>
    </>
  );
};
