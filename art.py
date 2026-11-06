"""
Backgammon with Expectiminimax AI - Artwork (Step 5)
Name:       Krishna Mahajan
Student ID: 106wucww
Module:     5FTC2067 Artificial Intelligence Principles - Mini Project

Every texture in the game is drawn in code (no image files are needed): wood grain, felt,
the board's points, ivory and ebony checkers, dice, gold lettering and the leather panels.
Each one is drawn once and cached, so the game only has to copy finished pictures every frame.
"""

import math
import random
import pygame

GOLD_LIGHT = (255, 236, 170)
GOLD = (222, 178, 88)
GOLD_DARK = (150, 104, 40)
IVORY_TEXT = (244, 236, 214)

_cache = {}        # finished pictures, keyed by what they show


def lerp(a, b, t):
    """
    Mixes two colours.
    Parameters:
      a: the colour at t = 0
      b: the colour at t = 1
      t: how far to go from a towards b (0..1)
    """
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def shade(colour, factor):
    """
    Makes a colour lighter (factor > 1) or darker (factor < 1).
    Parameters:
      colour: the RGB colour to change
      factor: brightness multiplier
    """
    return tuple(max(0, min(255, int(c * factor))) for c in colour)


def rounded_mask(w, h, radius):
    """
    A white rounded rectangle on a transparent surface, used to cut pictures to shape.
    Parameters:
      w: width in pixels
      h: height in pixels
      radius: corner radius in pixels
    """
    mask = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, w, h), border_radius=radius)
    return mask


def wood(w, h, base, grain, seed=1):
    """
    A plank of wood: a base colour with wavy grain lines, a few knots and soft lighting.
    Parameters:
      w: width in pixels
      h: height in pixels
      base: the wood's main colour
      grain: the darker colour of the grain lines
      seed: makes the same grain pattern every time
    """
    rng = random.Random(seed)
    surf = pygame.Surface((w, h))
    surf.fill(base)
    phase1, phase2 = rng.uniform(0, 6), rng.uniform(0, 6)
    for i in range(0, h, 2):
        colour = lerp(base, grain, rng.uniform(0.05, 0.55))
        amp, freq = rng.uniform(1, 4), rng.uniform(0.004, 0.012)
        pts = [(x, i + amp * math.sin(x * freq + phase1 + i * 0.03)
                + 1.5 * math.sin(x * 0.031 + phase2)) for x in range(0, w + 16, 16)]
        pygame.draw.lines(surf, colour, False, pts, rng.choice((1, 1, 2)))
    for _ in range(max(1, w * h // 120000)):          # a few knots
        kx, ky = rng.randint(0, w), rng.randint(0, h)
        for r in range(14, 2, -3):
            pygame.draw.ellipse(surf, lerp(base, grain, 0.3 + r / 30),
                                (kx - r * 2, ky - r // 2, r * 4, r), 1)
    light = pygame.Surface((w, h), pygame.SRCALPHA)
    for y in range(0, h, 4):                           # soft light from above, shade below
        v = 1 - 2 * y / max(1, h)                      # +1 at the top, 0 in the middle, -1 at the bottom
        if v > 0:
            light.fill((255, 240, 210, int(36 * v)), (0, y, w, 4))
        else:
            light.fill((0, 0, 0, int(-48 * v)), (0, y, w, 4))
    surf.blit(light, (0, 0))
    return surf


def felt(w, h, base, seed=2):
    """
    Playing felt: a fine random weave tiled across the area, darker towards the edges.
    Parameters:
      w: width in pixels
      h: height in pixels
      base: the felt's colour
      seed: makes the same weave every time
    """
    rng = random.Random(seed)
    tile = pygame.Surface((96, 96))
    tile.fill(base)
    for _ in range(2600):
        tile.set_at((rng.randrange(96), rng.randrange(96)), shade(base, rng.uniform(0.82, 1.18)))
    surf = pygame.Surface((w, h))
    for x in range(0, w, 96):
        for y in range(0, h, 96):
            surf.blit(tile, (x, y))
    edge = pygame.Surface((w, h), pygame.SRCALPHA)
    for k in range(28):                                # vignette: darkest at the very edge
        pygame.draw.rect(edge, (0, 0, 0, 2 + (28 - k) * 2), (k, k, w - 2 * k, h - 2 * k), 1)
    surf.blit(edge, (0, 0))
    return surf


def point(w, h, colour, down):
    """
    One triangular point of the board, with a lighter centre stripe and a thin gold edge.
    Parameters:
      w: width of the triangle's base in pixels
      h: height of the triangle in pixels
      colour: the point's colour
      down: True if the point hangs from the top edge, False if it rises from the bottom
    """
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    base_y, tip_y = (0, h) if down else (h, 0)
    for k in range(6):                                 # darker edges, lighter middle
        inset = k * w / 14
        c = shade(colour, 0.82 + k * 0.05)
        pygame.draw.polygon(surf, c, [(inset, base_y), (w - inset, base_y), (w / 2, tip_y)])
    edge_y = min(base_y, h - 1)                        # keep the base line inside the surface
    pygame.draw.aalines(surf, shade(colour, 0.55), True, [(0, edge_y), (w - 1, edge_y), (w / 2, tip_y)])
    pygame.draw.aaline(surf, GOLD_DARK, (1, edge_y), (w / 2, tip_y))
    pygame.draw.aaline(surf, GOLD_DARK, (w - 2, edge_y), (w / 2, tip_y))
    return surf


def checker(radius, white):
    """
    A checker seen from above: polished ivory or ebony with a groove, a rim and a highlight.
    Cached, so every checker on the board is the same picture.
    Parameters:
      radius: the checker's radius in pixels
      white: True for an ivory (White) checker, False for an ebony (Black) one
    """
    key = ("checker", radius, white)
    if key in _cache:
        return _cache[key]
    size = radius * 2 + 12
    c = size // 2
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    for k, alpha in ((6, 40), (4, 60), (2, 80)):       # soft shadow below and to the right
        pygame.draw.circle(surf, (0, 0, 0, alpha), (c + 3, c + 4), radius + k // 2)
    edge, centre = ((186, 170, 136), (252, 248, 236)) if white else ((10, 9, 10), (86, 78, 74))
    for r in range(radius, 0, -1):                     # light falls from the top left
        t = (1 - r / radius) ** 0.7
        off = int((1 - r / radius) * radius * 0.28)
        pygame.draw.circle(surf, lerp(edge, centre, t), (c - off, c - off), r)
    rim = shade(edge, 0.75) if white else (40, 36, 34)
    pygame.draw.circle(surf, rim, (c, c), radius, 2)
    groove = int(radius * 0.68)
    pygame.draw.circle(surf, shade(edge, 0.85) if white else (8, 8, 8), (c, c), groove, 2)
    pygame.draw.circle(surf, lerp(centre, (255, 255, 255), 0.4) if white else (110, 100, 96),
                       (c, c), groove + 2, 1)
    shine = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.ellipse(shine, (255, 255, 255, 70 if white else 45),
                        (c - radius * 0.75, c - radius * 0.85, radius * 0.9, radius * 0.5))
    surf.blit(shine, (0, 0))
    _cache[key] = surf
    return surf


def die(size, value, used=False):
    """
    A die face: rounded ivory with a gentle top highlight, an edge, a shadow and inset pips.
    Parameters:
      size: width and height of the face in pixels
      value: the number shown (1-6)
      used: True to grey it out (the die has been played)
    """
    key = ("die", size, value, used)
    if key in _cache:
        return _cache[key]
    pad = 6
    surf = pygame.Surface((size + pad, size + pad), pygame.SRCALPHA)
    radius = max(4, size // 6)
    pygame.draw.rect(surf, (0, 0, 0, 70), (4, 5, size, size), border_radius=radius)
    top, bottom = ((250, 248, 240), (214, 204, 182)) if not used else ((150, 146, 138), (110, 106, 100))
    body = pygame.Surface((size, size), pygame.SRCALPHA)
    for y in range(size):
        pygame.draw.line(body, lerp(top, bottom, y / size), (0, y), (size, y))
    body.blit(rounded_mask(size, size, radius), (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    surf.blit(body, (0, 0))
    pygame.draw.rect(surf, (120, 106, 86), (0, 0, size, size), 2, border_radius=radius)
    s, m = size // 4, size // 2
    spots = {1: [(m, m)], 2: [(s, s), (3 * s, 3 * s)], 3: [(s, s), (m, m), (3 * s, 3 * s)],
             4: [(s, s), (3 * s, s), (s, 3 * s), (3 * s, 3 * s)],
             5: [(s, s), (3 * s, s), (m, m), (s, 3 * s), (3 * s, 3 * s)],
             6: [(s, s), (3 * s, s), (s, m), (3 * s, m), (s, 3 * s), (3 * s, 3 * s)]}
    pip = max(2, size // 11)
    for x, y in spots[value]:
        pygame.draw.circle(surf, (24, 20, 18), (x, y), pip)
        pygame.draw.circle(surf, (80, 70, 62), (x - pip // 3, y - pip // 3), max(1, pip // 3))
    _cache[key] = surf
    return surf


def cube_face(size, label, font):
    """
    The doubling cube: an ivory block like a die, with its value in dark gold.
    Parameters:
      size: width and height in pixels
      label: the number shown ("64" while the cube is still at 1)
      font: the pygame font for the number
    """
    key = ("cube", size, label, font.get_height())
    if key in _cache:
        return _cache[key]
    surf = die(size, 1).copy()
    pygame.draw.rect(surf, (238, 232, 216), (size // 2 - size // 6, size // 2 - size // 6,
                                             size // 3, size // 3))       # cover the single pip
    t = font.render(label, True, (122, 82, 26))
    surf.blit(t, t.get_rect(center=(size // 2, size // 2 + 1)))
    _cache[key] = surf
    return surf


def tray(w, h, felt_colour, seed=5):
    """
    The bear-off tray: a recessed strip of darker felt with a shadow along its top and left edges.
    Parameters:
      w: width in pixels
      h: height in pixels
      felt_colour: colour of the board's felt (the tray is a darker shade of it)
      seed: weave pattern
    """
    surf = felt(w, h, shade(felt_colour, 0.55), seed)
    edge = pygame.Surface((w, h), pygame.SRCALPHA)
    for k in range(8):
        pygame.draw.line(edge, (0, 0, 0, 90 - k * 10), (0, k), (w, k))
        pygame.draw.line(edge, (0, 0, 0, 90 - k * 10), (k, 0), (k, h))
    surf.blit(edge, (0, 0))
    return surf


def gold_text(font, text, spacing=0):
    """
    Lettering in engraved gold: a gold gradient with a dark drop shadow and a light top edge.
    Parameters:
      font: the pygame font to use
      text: the words to draw
      spacing: extra pixels between letters (for a wide, carved look)
    """
    key = ("gold", font.get_height(), text, spacing)
    if key in _cache:
        return _cache[key]
    if spacing:
        letters = [font.render(ch, True, (255, 255, 255)) for ch in text]
        w = sum(l.get_width() for l in letters) + spacing * (len(text) - 1)
        mask = pygame.Surface((w, font.get_height()), pygame.SRCALPHA)
        x = 0
        for l in letters:
            mask.blit(l, (x, 0))
            x += l.get_width() + spacing
    else:
        mask = font.render(text, True, (255, 255, 255))
    w, h = mask.get_size()
    grad = pygame.Surface((w, h), pygame.SRCALPHA)
    for y in range(h):                                 # light top, deep middle, warm bottom
        t = y / max(1, h - 1)
        colour = lerp(GOLD_LIGHT, GOLD_DARK, t / 0.55) if t < 0.55 else lerp(GOLD_DARK, GOLD, (t - 0.55) / 0.45)
        pygame.draw.line(grad, colour + (255,), (0, y), (w, y))
    gold = mask.copy()
    gold.blit(grad, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    out = pygame.Surface((w + 6, h + 6), pygame.SRCALPHA)
    shadow = mask.copy()
    shadow.fill((30, 16, 6, 200), special_flags=pygame.BLEND_RGBA_MULT)
    out.blit(shadow, (3, 4))
    edge = mask.copy()
    edge.fill((255, 250, 225, 150), special_flags=pygame.BLEND_RGBA_MULT)
    out.blit(edge, (1, 0))
    out.blit(gold, (2, 2))
    _cache[key] = (out, mask)
    return out, mask


def plain_text(font, text, colour=IVORY_TEXT, shadow=True):
    """
    Text with a soft dark shadow, so it stays readable on wood and felt.
    Parameters:
      font: the pygame font to use
      text: the words to draw
      colour: the text colour
      shadow: False to draw the text without a shadow
    """
    key = ("text", font.get_height(), text, colour, shadow)
    if key in _cache:
        return _cache[key]
    img = font.render(text, True, colour)
    if shadow:
        out = pygame.Surface((img.get_width() + 2, img.get_height() + 2), pygame.SRCALPHA)
        out.blit(font.render(text, True, (10, 6, 4)), (2, 2))
        out.blit(img, (0, 0))
        img = out
    if len(_cache) < 4000:                             # labels repeat; long logs are not cached forever
        _cache[key] = img
    return img


def panel(w, h, radius=14, seed=9):
    """
    A dark walnut panel with a double gold border, used for the side panel and dialogs.
    Parameters:
      w: width in pixels
      h: height in pixels
      radius: corner radius in pixels
      seed: grain pattern
    """
    key = ("panel", w, h, radius, seed)
    if key in _cache:
        return _cache[key]
    surf = wood(w, h, (54, 34, 22), (30, 18, 10), seed)
    dark = pygame.Surface((w, h), pygame.SRCALPHA)
    dark.fill((0, 0, 0, 90))
    surf.blit(dark, (0, 0))
    out = rounded_mask(w, h, radius)
    out.blit(surf, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    pygame.draw.rect(out, GOLD, (0, 0, w, h), 2, border_radius=radius)
    pygame.draw.rect(out, GOLD_DARK, (6, 6, w - 12, h - 12), 1, border_radius=max(2, radius - 5))
    _cache[key] = out
    return out


def button(surf, rect, colour, label, hover, enabled):
    """
    Draws a jewel-like button: a rounded gradient in the button's colour, a gold edge,
    a highlight along the top and a shadow underneath.
    Parameters:
      surf: the surface to draw on
      rect: the button's pygame Rect
      colour: the button's main colour
      label: the rendered text to put in the middle
      hover: True if the mouse is over the button
      enabled: False to draw it greyed out
    """
    if not enabled:
        colour = (74, 66, 60)
    elif hover:
        colour = shade(colour, 1.18)
    key = ("button", rect.size, colour, enabled)
    if key not in _cache:                              # drawn once per size, colour and state
        w, h = rect.size
        body = pygame.Surface((w, h + 3), pygame.SRCALPHA)
        pygame.draw.rect(body, (0, 0, 0, 90), (0, 3, w, h), border_radius=10)      # shadow
        face = pygame.Surface((w, h), pygame.SRCALPHA)
        top, bottom = shade(colour, 1.25), shade(colour, 0.7)
        for y in range(h):
            pygame.draw.line(face, lerp(top, bottom, y / max(1, h)), (0, y), (w, y))
        face.blit(rounded_mask(w, h, 10), (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        body.blit(face, (0, 0))
        pygame.draw.rect(body, GOLD if enabled else (110, 96, 76), (0, 0, w, h), 2, border_radius=10)
        pygame.draw.line(body, shade(colour, 1.5), (10, 3), (w - 10, 3))
        _cache[key] = body
    surf.blit(_cache[key], rect.topleft)
    surf.blit(label, label.get_rect(center=(rect.centerx, rect.centery + 1)))


def ornament(surf, centre, half_width, colour=GOLD):
    """
    A thin gold rule with a diamond in the middle, used under titles.
    Parameters:
      surf: the surface to draw on
      centre: (x, y) of the diamond
      half_width: length of each side of the rule in pixels
      colour: line colour
    """
    x, y = centre
    pygame.draw.line(surf, colour, (x - half_width, y), (x - 12, y), 2)
    pygame.draw.line(surf, colour, (x + 12, y), (x + half_width, y), 2)
    pygame.draw.polygon(surf, colour, [(x, y - 7), (x + 8, y), (x, y + 7), (x - 8, y)])
    for side in (-1, 1):
        pygame.draw.circle(surf, colour, (x + side * (half_width + 6), y), 3)
