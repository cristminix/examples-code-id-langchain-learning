import { HumanMessage, SystemMessage } from "langchain"
import { createChatModel } from "./createChatModel"

export async function chatModelBasic() {
  const model = createChatModel()
  const prompt = [new HumanMessage("What is the capital of France?")]

  const result = await model.invoke(prompt)
  console.log(result.content)
}

export async function chatModelWithSystemInstruction() {
  const model = createChatModel()
  const humanMsg = new HumanMessage("What is the capital of France?")
  const systemMsg =
    new SystemMessage(`You are a helpful assistant that responds to questions with three 
      exclamation marks.`)
  const prompt = [systemMsg, humanMsg]

  const result = await model.invoke(prompt)
  console.log(result.content)
}
