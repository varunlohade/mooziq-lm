// Every style, by id, in the order the picker shows them.
import * as jazzhop from '../jazzhop/compose.js';
import * as flow from '../flow/compose.js';
import * as steady from '../steady/music.js';
import * as rainycafe from './rainycafe.js';
import * as sleep from './sleep.js';
import * as studypiano from './studypiano.js';
import * as chillhop from './chillhop.js';
import * as nature from './nature.js';
export const STYLES = {steady, flow, sleep, studypiano, nature, rainycafe, chillhop, jazzhop};
export const ORDER = Object.keys(STYLES);
export const NAMES = {steady: 'Steady', flow: 'Flow', sleep: 'Sleep', studypiano: 'Study Piano', nature: 'Nature', rainycafe: 'Rainy Café', chillhop: 'Chillhop', jazzhop: 'jazz(experimental)'};
