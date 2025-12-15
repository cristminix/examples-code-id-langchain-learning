import { PromptTemplate } from "@langchain/core/prompts"
import { HumanMessage, SystemMessage } from "@langchain/core/messages"
import { ChatOpenAI } from "@langchain/openai"
import { config } from "dotenv"
import { ChatPromptTemplate } from "@langchain/core/prompts"

config()
export const main = async () => {
  const model = new ChatOpenAI({
    model: process.env.OPENAI_MODEL,
    temperature: 0.5,
    // maxTokens: 100,
  })

  const template = ChatPromptTemplate.fromMessages([
    [
      "system",
      `Answer the question based on the context below. If the question 
    cannot be answered using the information provided, answer with "I 
    don\'t know".`,
    ],
    ["human", "Context: {context}"],
    ["human", "Question: {question}"],
  ])
  console.log(template)
}
main().catch((e) => console.error(e))
