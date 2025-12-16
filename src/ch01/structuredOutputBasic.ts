import { z } from "zod"
import { createChatModel } from "./createChatModel"
import { CommaSeparatedListOutputParser } from "@langchain/core/output_parsers"
export async function structuredOutputBasic() {
  // Mendefinisikan skema respons terstruktur menggunakan Zod.
  const answerSchema = z.object({
    question: z.string().describe("Pertanyaan pengguna"),

    // Bidang 'answer': Jawaban atas pertanyaan pengguna.
    answer: z.string().describe("Jawaban atas pertanyaan pengguna"),
    // Bidang 'justification': Pembenaran/alasan untuk jawaban tersebut.
    justification: z.string().describe(`Pembenaran/alasan untuk
      jawaban`),
    // Deskripsi keseluruhan skema: Jawaban beserta pembenarannya.
  }).describe(`Jawaban atas pertanyaan pengguna beserta pembenaran/alasan untuk
    jawaban tersebut.`)

  // Membuat model obrolan dan mengkonfigurasinya untuk output terstruktur berdasarkan skema.
  const model = createChatModel().withStructuredOutput(answerSchema)

  // Memanggil model dengan pertanyaan.
  const result = await model.invoke(
    "Apa yang lebih berat, satu pon batu bata atau satu pon bulu" // Pertanyaan yang diterjemahkan
  )

  // Mencetak hasilnya (yang akan sesuai dengan skema).
  console.log(result)
}

export async function commaSeparatedBasic() {
  const parser = new CommaSeparatedListOutputParser()

  const result = await parser.invoke("apple, banana, cherry")

  console.log(result)
}
