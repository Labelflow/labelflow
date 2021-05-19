import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import "@testing-library/jest-dom/extend-expect";

import { ImportImagesModal } from "../import-images-modal";

test("should return the list of images the user picked", async () => {
  const file = new File(["Hello"], "hello.png", { type: "image/png" });
  const onImportSucceed = jest.fn();

  render(<ImportImagesModal onImportSucceed={onImportSucceed} />);
  const input = screen.getByLabelText(/drop folders or images/i);
  await waitFor(() => userEvent.upload(input, file));

  expect(onImportSucceed).toHaveBeenCalledWith([file]);
});

test("should display the number of images", async () => {
  const file = new File(["Hello"], "hello.png", { type: "image/png" });
  const onImportSucceed = jest.fn();

  render(<ImportImagesModal onImportSucceed={onImportSucceed} />);
  const input = screen.getByLabelText(/drop folders or images/i);
  await waitFor(() => userEvent.upload(input, file));

  expect(screen.getByText(/uploading 1 items/i)).toBeDefined();
  expect(
    screen.queryByLabelText(/drop folders or images/i)
  ).not.toBeInTheDocument();
});
