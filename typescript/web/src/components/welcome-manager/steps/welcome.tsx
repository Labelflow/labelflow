import { useEffect, useState, useRef } from "react";
import {
  chakra,
  ModalContent,
  ModalFooter,
  VStack,
  Box,
  Stack,
  Button,
  Heading,
  Center,
  Text,
  ModalBody,
  ModalHeader,
  useColorModeValue as mode,
} from "@chakra-ui/react";
import { RiPlayFill, RiSpeedMiniFill } from "react-icons/ri";

import useCountDown from "react-countdown-hook";

import { Logo } from "../../logo";

const PlayIcon = chakra(RiPlayFill);
const SpeedIcon = chakra(RiSpeedMiniFill);

type Props = {
  onClickNext: () => void;
  onClickSkip?: () => void;
  autoStartCountDown?: boolean;
};

const delayMilliSeconds = 10 * 1000;

export const Welcome = ({
  onClickSkip,
  onClickNext,
  autoStartCountDown = true,
}: Props) => {
  const [isCountDownStarted, setIsCountDownStarted] = useState(false);
  const [timeLeft, { start }] = useCountDown(delayMilliSeconds, 1000 / 30);
  const startLabelingButtonRef = useRef<HTMLButtonElement>(null);
  // Start the timer during the first render
  useEffect(() => {
    if (autoStartCountDown) {
      setIsCountDownStarted(true);
      start();
      startLabelingButtonRef.current?.focus();
    }
  }, []);

  useEffect(() => {
    if (isCountDownStarted && timeLeft <= 50 && onClickNext) {
      onClickNext();
    }
  }, [isCountDownStarted, timeLeft]);

  return (
    <ModalContent margin={{ base: "4", md: "3.75rem" }}>
      <ModalHeader textAlign="center" padding={{ base: "4", md: "8" }}>
        <Center flexDirection="column">
          <Logo maxW="lg" my={{ base: "2", md: "8" }} h="min-content" />
        </Center>
      </ModalHeader>

      <ModalBody>
        <VStack
          justifyContent="space-evenly"
          spacing="8"
          h="full"
          mt="0"
          mb={{ base: "4", md: "8" }}
        >
          <Heading
            as="h1"
            size="2xl"
            maxW="lg"
            color={mode("gray.600", "gray.300")}
            fontWeight="extrabold"
            textAlign="center"
          >
            The open standard{" "}
            <Text color="brand.500" display="inline">
              image labeling tool
            </Text>
          </Heading>
          <Text
            color={mode("gray.600", "gray.400")}
            mt="16"
            maxW="lg"
            fontSize="lg"
            fontWeight="medium"
            textAlign="justify"
          >
            Stay in control of your data, label your images fast, without them
            leaving your computer. Focus on building the next big thing while we
            ensure privacy and performance.
            <br />
            <br />
            The tutorial will start automatically in{" "}
            {Math.round(timeLeft / 1000)} seconds.
          </Text>
        </VStack>
      </ModalBody>

      <ModalFooter>
        <Stack
          direction={{ base: "column", md: "row" }}
          justifyContent="center"
          width="full"
          spacing="4"
          mb={{ base: "0", md: "10" }}
        >
          <Button
            size="lg"
            leftIcon={<SpeedIcon fontSize="xl" />}
            minW="210px"
            variant="link"
            height="14"
            px="8"
            loadingText="Skip the tutorial"
            onClick={onClickSkip}
          >
            Skip the tutorial
          </Button>

          <Button
            ref={startLabelingButtonRef}
            leftIcon={<PlayIcon fontSize="xl" />}
            size="lg"
            minW="210px"
            colorScheme="brand"
            height="14"
            px="8"
            loadingText="Get started!"
            onClick={onClickNext}
            position="relative"
          >
            Get started!
            <Box
              position="absolute"
              bottom="0"
              left="0"
              height="2"
              borderBottomLeftRadius={4}
              width={`${(100 * timeLeft) / delayMilliSeconds}%`}
              background="#ffffff88"
            />
          </Button>
        </Stack>
      </ModalFooter>
    </ModalContent>
  );
};
