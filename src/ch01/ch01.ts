import { config } from "dotenv"
import {
  chatModelBasic,
  chatModelWithSystemInstruction,
} from "./chatModelBasic"
import {
  chatPromptTemplateBasic,
  promptTemplateBasic,
} from "./promptTemplateBasic"
import {
  commaSeparatedBasic,
  structuredOutputBasic,
} from "./structuredOutputBasic"
import { runableBasic } from "./runableBasic"
import {
  chainBasic,
  chainBasic2,
  compositionBasic,
  compositionBasic2,
} from "./compositionBasic"
config()
export const main = async () => {
  // await chatModelBasic()
  // await chatModelWithSystemInstruction()
  // await promptTemplateBasic()
  // await chatPromptTemplateBasic()
  // await structuredOutputBasic()
  // await commaSeparatedBasic()
  // await runableBasic()
  // await compositionBasic()
  // await compositionBasic2()
  // await chainBasic()
  await chainBasic2()
}
main().catch((e) => console.error(e))
