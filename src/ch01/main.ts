import { HumanMessage, SystemMessage } from "@langchain/core/messages"
import { ChatOpenAI } from "@langchain/openai"
import { config } from "dotenv"
import { createChatModel } from "./createChatModel"
config()
export const main = async () => {
  const model = createChatModel()

  //   const result = await model.invoke("The sky is")
  const prompt = [
    new SystemMessage(
      `You are a helpful assistant that responds to questions with three 
      exclamation marks.`
    ),
    new HumanMessage("What is the capital of France?"),
  ]

  const result = await model.invoke(prompt)
  console.log(result.content)
}
main().catch((e) => console.error(e))
