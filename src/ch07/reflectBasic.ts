import {
  AIMessage,
  BaseMessage,
  SystemMessage,
  HumanMessage,
} from "@langchain/core/messages";
import {
  StateGraph,
  Annotation,
  messagesStateReducer,
  START,
  END,
} from "@langchain/langgraph";
import { createChatModel } from "../ch01/createChatModel";

export async function reflectBasic() {
  const model = createChatModel();
  const annotation = Annotation.Root({
    messages: Annotation({ reducer: messagesStateReducer, default: () => [] }),
  });

  // perbaiki string multi-baris
  const generatePrompt = new SystemMessage(
    `Anda adalah asisten esai yang ditugaskan menulis esai 3-paragraf yang sangat baik.
    Hasilkan esai terbaik yang mungkin untuk permintaan pengguna.
    Jika pengguna memberikan kritik, tanggapi dengan versi revisi dari
      upaya Anda sebelumnya.`,
  );

  async function generate(state) {
    const answer = await model.invoke([generatePrompt, ...state.messages]);
    return { messages: [answer] };
  }

  const reflectionPrompt = new SystemMessage(
    `Anda adalah guru yang menilai pengajuan esai. Hasilkan kritik dan
      rekomendasi untuk pengajuan pengguna.
    Berikan rekomendasi terperinci, termasuk permintaan untuk panjang, kedalaman,
      gaya, dll.`,
  );

  async function reflect(state) {
    // Balikkan pesan untuk membuat LLM berefleksi pada keluaran sendiri
    const clsMap: { [key: string]: new (content: string) => BaseMessage } = {
      ai: HumanMessage,
      human: AIMessage,
    };
    // Pesan pertama adalah permintaan pengguna asli.
    // kita pertahankan sama untuk semua simpul
    const translated = [
      reflectionPrompt,
      state.messages[0],
      ...state.messages
        .slice(1)
        .map((msg) => new clsMap[msg._getType()](msg.content as string)),
    ];
    const answer = await model.invoke(translated);
    // kita perlukan keluaran ini sebagai umpan balik manusia untuk generator
    return { messages: [new HumanMessage({ content: answer.content })] };
  }

  function shouldContinue(state) {
    if (state.messages.length > 6) {
      // Berhenti setelah 3 iterasi, masing-masing dengan 2 pesan
      return END;
    } else {
      return "reflect";
    }
  }

  const builder = new StateGraph(annotation)
    .addNode("generate", generate)
    .addNode("reflect", reflect)
    .addEdge(START, "generate")
    .addConditionalEdges("generate", shouldContinue)
    .addEdge("reflect", "generate");

  const graph = builder.compile();
  // await graph.getGraph().drawMermaidPng()
  const input = {
    messages: [new HumanMessage(`Berdamai dengan yang tidak kita suka`)],
  };
  for await (const c of await graph.stream(input)) {
    const keys = Object.keys(c);
    const [key] = keys;
    const { messages } = c[key];
    for (const message of messages) {
      console.log(message.content);
    }
  }
}
