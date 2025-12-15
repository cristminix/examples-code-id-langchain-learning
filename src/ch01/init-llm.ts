import { ChatOpenAI } from "@langchain/openai"
import { config } from "dotenv"
config()
export const main = async () => {
  const model = new ChatOpenAI({ model: process.env.OPENAI_MODEL })

  const result = await model.invoke("The sky is")
  console.log(result.content)
}
main().catch((e) => console.error(e))
