import { config } from "dotenv"
import {
  chatModelBasic,
  chatModelWithSystemInstruction,
} from "./chatModelBasic"
import {
  chatPromptTemplateBasic,
  promptTemplateBasic,
} from "./promptTemplateBasic"
import { structuredOutputBasic } from "./structuredOutputBasic"
config()
export const main = async () => {
  // await chatModelBasic()
  // await chatModelWithSystemInstruction()
  // await promptTemplateBasic()
  // await chatPromptTemplateBasic()
  await structuredOutputBasic()
}
main().catch((e) => console.error(e))
