import { ChatOpenAI } from "@langchain/openai"
import { StructuredOutputParser } from "@langchain/core/output_parsers"

// const originalInvoke = ChatOpenAI.prototype.invoke
const originalWithStructuredOutput = ChatOpenAI.prototype.withStructuredOutput

// Helper to convert Zod schema to description
function getSchemaDescription(schema) {
  const parser = StructuredOutputParser.fromZodSchema(schema)

  return parser.getFormatInstructions()
}
// Alternative: Patch the withStructuredOutput method itself
// @ts-ignore
ChatOpenAI.prototype.withStructuredOutput = function (schema, config) {
  const structuredModel = originalWithStructuredOutput.call(
    this,
    schema,
    config
  )

  // Get the schema description
  const schemaDescription = getSchemaDescription(schema)

  const originalInvoke = structuredModel.invoke
  structuredModel.invoke = function (input) {
    if (typeof input === "string") {
      const enhancedPrompt = `${schemaDescription}

 ${input}`

      return originalInvoke.call(this, enhancedPrompt)
    }
    return originalInvoke.call(this, input)
  }

  return structuredModel
}
export function createChatModel(options: any = {}) {
  const realOptions = {
    model: process.env.OPENAI_MODEL,
    temperature: 0.5,
    ...options,
  }
  return new ChatOpenAI(realOptions)
}
