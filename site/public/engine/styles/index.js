// Every style, by id, in the order the picker shows them.
import * as jazzhop from '../jazzhop/compose.js';
import * as flow from '../flow/compose.js';
import * as steady from '../steady/music.js';
import * as rainycafe from './rainycafe.js';
import * as sleep from './sleep.js';
import * as bossa from './bossa.js';
import * as studypiano from './studypiano.js';
import * as chillhop from './chillhop.js';
import * as nature from './nature.js';
export const STYLES = {jazzhop, flow, steady, rainycafe, sleep, bossa, studypiano, chillhop, nature};
export const ORDER = Object.keys(STYLES);
export const NAMES = {jazzhop: 'Jazz Hop', flow: 'Flow', steady: 'Steady', rainycafe: 'Rainy Café', sleep: 'Sleep', bossa: 'Bossa',
  studypiano: 'Study Piano', chillhop: 'Chillhop', nature: 'Nature'};
