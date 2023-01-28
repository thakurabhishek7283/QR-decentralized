import Web3 from "web3";

let web3;
const web3Instance = async () => {
  if (typeof window !== "undefined" && typeof window.ethereum !== "undefined") {
    web3 = new Web3(window.ethereum);
    await window.ethereum.enable();
  } else if (
    typeof window !== "undefined" &&
    typeof window.web3 !== "undefined"
  ) {
    web3 = new Web3(window.web3.currentProvider);
  } else {
    const provider = new Web3.providers.HttpProvider(
      "wss://2Kg3XO3YdZBgde1ZM5o9Qm3pHud:ab321cecb64147decf5c15367e4f9602@filecoin.infura.io"
    );
    web3 = new Web3(provider);
  }
};
web3Instance();

export default web3;
