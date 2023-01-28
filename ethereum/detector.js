import web3 from "./web3";
import FakeProductDetector from "../ethereum/build/FakeProductDetector.json";

const instance = new web3.eth.Contract(
  JSON.parse(FakeProductDetector.interface),
  "Address of contract"
);

export default instance;
