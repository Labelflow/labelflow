/* eslint-disable import/first */
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MockedProvider } from "@apollo/client/testing";
import { PropsWithChildren } from "react";
import "@testing-library/jest-dom/extend-expect";
import {
  mockUseQueryParams,
  mockNextRouter,
} from "../../../../utils/router-mocks";
import { BASIC_DATASET_DATA } from "../../../../utils/tests/data.fixtures";

mockUseQueryParams();
mockNextRouter({
  isReady: true,
  query: {
    datasetSlug: BASIC_DATASET_DATA.slug,
    workspaceSlug: BASIC_DATASET_DATA.workspace.slug,
  },
});

import { ImportImagesModal } from "../import-images-modal";
import { getApolloMockLink } from "../../../../utils/tests/apollo-mock";
import { ERROR_MOCKS, IMPORT_BUTTON_MOCKS } from "../../import-button.fixtures";

const files = [
  new File(["Hello"], "hello.png", { type: "image/png" }),
  new File(["World"], "world.png", { type: "image/png" }),
  new File(["Error"], "error.pdf", { type: "application/pdf" }),
];

const createWrapper =
  (error: boolean = false) =>
  ({ children }: PropsWithChildren<{}>) =>
    (
      <MockedProvider
        link={getApolloMockLink(error ? ERROR_MOCKS : IMPORT_BUTTON_MOCKS)}
      >
        {children}
      </MockedProvider>
    );

async function ensuresUploadsAreFinished(number = 2) {
  await waitFor(() =>
    expect(screen.getAllByLabelText("Upload succeed")).toHaveLength(number)
  );
}

const renderModalAndImport = (
  filesToImport = files,
  props = {},
  error = false
) => {
  render(<ImportImagesModal isOpen onClose={() => {}} {...props} />, {
    wrapper: createWrapper(error),
  });

  const input = screen.getByLabelText(/drop folders or images/i);
  return waitFor(() => userEvent.upload(input, filesToImport));
};

test("should display the number of valid images", async () => {
  await renderModalAndImport();

  await waitFor(() =>
    expect(screen.getByText(/Completed 2 of 2 items/i)).toBeDefined()
  );
  expect(
    screen.queryByLabelText(/drop folders or images/i)
  ).not.toBeInTheDocument();
});

test("should display an indicator when upload succeed", async () => {
  await renderModalAndImport(files.slice(0, 1));

  await waitFor(() =>
    expect(screen.getByLabelText("Upload succeed")).toBeDefined()
  );
});

test("should display an indicator when upload failed", async () => {
  await renderModalAndImport(files.slice(0, 1), {}, true);

  await waitFor(() =>
    expect(screen.getByLabelText("Error indicator")).toBeDefined()
  );
});

test("should display a loading indicator when file is uploading", async () => {
  await renderModalAndImport(files.slice(0, 1));

  await waitFor(() =>
    expect(screen.getByLabelText("Loading indicator")).toBeDefined()
  );
  await waitFor(() =>
    expect(screen.getByLabelText("Upload succeed")).toBeDefined()
  );
});

test("when the user drags invalid formats, only the valid pictures are uploaded", async () => {
  await renderModalAndImport();

  await waitFor(() =>
    expect(screen.getAllByLabelText("Upload succeed")).toHaveLength(2)
  );
});

test("should display the images name", async () => {
  await renderModalAndImport();

  expect(screen.getByText(/hello.png/i)).toBeDefined();
  expect(screen.getByText(/world.png/i)).toBeDefined();

  await ensuresUploadsAreFinished();
});

test("should display the rejected images name", async () => {
  await renderModalAndImport(files.slice(2, 3));

  expect(screen.getByText(/error.pdf/i)).toBeDefined();
  expect(screen.getByText(/file type must be jpeg, png or bmp/i)).toBeDefined();
});

test("should display the error description when a file could not be imported", async () => {
  await renderModalAndImport(files.slice(2, 3));

  expect(screen.getByText(/file type must be jpeg, png or bmp/i)).toBeDefined();

  userEvent.hover(screen.getByText(/file type must be jpeg, png or bmp/i));

  // We need to wait for the tooltip to be rendered before checking its content.
  await waitFor(() =>
    expect(screen.getByText(/File type must be/i)).toBeDefined()
  );
});

test("should not display the modal by default", async () => {
  act(() => {
    render(<ImportImagesModal />, {
      wrapper: createWrapper(),
    });
  });

  expect(screen.queryByText(/Import/i)).not.toBeInTheDocument();
});

test("should call the onClose handler", async () => {
  const onClose = jest.fn();
  await renderModalAndImport([], { onClose });

  userEvent.click(screen.getByLabelText("Close"));

  expect(onClose).toHaveBeenCalled();
});

test("should not close the modal while file are uploading", async () => {
  await renderModalAndImport(files.slice(0, 1));

  expect(screen.getByLabelText("Loading indicator")).toBeDefined();
  expect(screen.getByLabelText("Close")).toBeDisabled();

  await ensuresUploadsAreFinished(1);
});

test("should display a start labeling button only when all the files are done", async () => {
  await renderModalAndImport(files);

  expect(
    screen.queryByRole("button", { name: /Start labeling/ })
  ).not.toBeInTheDocument();

  await waitFor(() =>
    expect(screen.getByRole("button", { name: /Start labeling/ })).toBeDefined()
  );
});
