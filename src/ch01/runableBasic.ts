import { createChatModel } from "./createChatModel"
export async function runableBasic() {
  const model = createChatModel()
  const completion = await model.invoke("Hi there!")
  console.log({ completion: completion.content })

  const completions = await model.batch(["Hi there!", "Bye!"])
  completions.forEach((completion) =>
    console.log({ completion: completion.content })
  )

  for await (const token of await model.stream("Bye!")) {
    console.log({ token: token.content })
  }
}
