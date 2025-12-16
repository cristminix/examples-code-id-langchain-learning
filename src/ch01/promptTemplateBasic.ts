import { ChatPromptTemplate, PromptTemplate } from "@langchain/core/prompts"
import { createChatModel } from "./createChatModel"
const context = `Kemajuan terbaru dalam NLP didorong oleh Large Language Models
(LLM). Model-model ini mengungguli model-model yang lebih kecil dan telah
menjadi sangat berharga bagi para pengembang yang membuat aplikasi dengan
kemampuan NLP. Pengembang dapat mengakses model-model ini melalui library
\`transformers\` dari Hugging Face, atau dengan memanfaatkan layanan dari OpenAI
dan Cohere melalui library \`openai\` dan \`cohere\`.`
const question = "Penyedia model mana saja yang menawarkan LLM?"

export const promptTemplateBasic = async () => {
  const model = createChatModel()

  const template =
    PromptTemplate.fromTemplate(`Answer the question based on the 
  context below. If the question cannot be answered using the information 
  provided, answer with "I don't know".

Context: {context}

Question: {question}

Answer: `)
  const inputPrompt = await template.format({ context, question })
  const result = await model.invoke(inputPrompt)
  console.log(result.content)
}
export const chatPromptTemplateBasic = async () => {
  const model = createChatModel()

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
  const inputPrompt = await template.format({ context, question })
  const result = await model.invoke(inputPrompt)
  console.log(result.content)
}
