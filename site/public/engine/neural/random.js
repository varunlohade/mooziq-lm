export function random(seed) {
  let state=seed>>>0;
  return () => {
    state=(state+0x6d2b79f5)>>>0;
    let x=Math.imul(state^(state>>>15),state|1);
    x^=x+Math.imul(x^(x>>>7),x|61);
    return ((x^(x>>>14))>>>0)/4294967296;
  };
}
