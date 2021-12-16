import { client } from "../../typescript/web/src/connectors/apollo-client/schema-client";
import { createDataset } from "./graphql-definitions.common";
import { declareImageNavigationTests } from "./image-navigation.common";

describe("Image Navigation (local)", () => {
  let datasetSlug: string;
  beforeEach(() => {
    cy.setCookie("consentedCookies", "true");
    cy.window().then(async () => {
      const { slug } = await createDataset({
        name: "cypress test dataset",
        workspaceSlug: "local",
        client,
      });
      datasetSlug = slug;
    });
  });

  declareImageNavigationTests({
    workspaceSlug: "local",
    getDatasetSlug: () => datasetSlug,
  });
});