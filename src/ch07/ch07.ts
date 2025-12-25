import { config } from "dotenv";
import { reflectBasic } from "./reflectBasic";
config();
export const main = async () => {
  await reflectBasic();
};
main().catch((e) => console.error(e));
