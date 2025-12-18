import { ChatPromptTemplate } from "@langchain/core/prompts"
import { createChatModel } from "./createChatModel"
import { RunnableLambda } from "@langchain/core/runnables"
export async function compositionBasic() {
  const model = createChatModel()

  const template = ChatPromptTemplate.fromMessages([
    ["system", "You are a helpful assistant."],
    ["human", "{question}"],
  ])

  // combine them in a function
  // RunnableLambda adds the same Runnable interface for any function you write

  const chatbot = RunnableLambda.from(async (values) => {
    const prompt = await template.invoke(values)
    return await model.invoke(prompt)
  })

  // use it

  const result = await chatbot.invoke({
    question: "Which model providers offer LLMs?",
  })

  console.log(result.content)
}

export async function compositionBasic2() {
  const model = createChatModel()

  const template = ChatPromptTemplate.fromMessages([
    ["system", "You are a helpful assistant."],
    ["human", "{question}"],
  ])

  const chatbot = RunnableLambda.from(async function* (values) {
    const prompt = await template.invoke(values)
    for await (const token of await model.stream(prompt)) {
      yield token
    }
  })

  for await (const token of await chatbot.stream({
    question: "Which model providers offer LLMs?",
  })) {
    process.stdout.write((token as any).content)
  }
  console.log("")
}
export async function chainBasic() {
  const model = createChatModel()

  const template = ChatPromptTemplate.fromMessages([
    ["system", "You are a helpful assistant."],
    ["human", "{question}"],
  ])

  // combine them in a function
  const chatbot = template.pipe(model)

  // use it

  const result = await chatbot.invoke({
    question: "Which model providers offer LLMs?",
  })

  console.log(result.content)
}
export async function chainBasic2() {
  const model = createChatModel()

  const template = ChatPromptTemplate.fromMessages([
    ["system", "You are a helpful assistant."],
    ["human", "{question}"],
  ])

  const chatbot = template.pipe(model)

  for await (const token of await chatbot.stream({
    question: "Apa yang dimaksud dengan apem tembem?",
  })) {
    process.stdout.write((token as any).content)
  }
  console.log("")
}
