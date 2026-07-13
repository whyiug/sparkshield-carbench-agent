"""Goal-specific voice responses for deterministic gate outcomes."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from typing import Any

from ..domain import GateOutcome
from .tts_guard import TTSGuard


def _language(language: str) -> str:
    normalized = language.lower().replace("_", "-")
    if normalized.startswith("zh") or normalized in {"chinese", "mandarin"}:
        return "zh"
    if normalized.startswith("de") or normalized == "german":
        return "de"
    return "en"


class ResponseComposer:
    def __init__(self, guard: TTSGuard | None = None) -> None:
        self.guard = guard or TTSGuard()

    def limitation(
        self,
        *,
        goal: str,
        reason: str,
        outcome: GateOutcome,
        language: str = "en",
    ) -> str:
        lang = _language(language)
        evidence = outcome is GateOutcome.UNAVAILABLE_EVIDENCE
        if lang == "zh":
            verb = "无法安全地" if evidence else "目前无法"
            text = f"我{verb}{goal}，因为{reason}。我没有执行这项操作。"
        elif lang == "de":
            qualifier = "sicher " if evidence else ""
            text = (
                f"Ich kann {goal} derzeit nicht {qualifier}ausfuehren, weil {reason}. "
                "Ich habe die Aktion nicht ausgefuehrt."
            )
        else:
            qualifier = " safely" if evidence else ""
            if reason == "the requested state change could not be verified":
                reason = "I couldn't verify the requested state change"
            text = (
                f"I can't{qualifier} {goal} because {reason}. "
                "I haven't performed that action."
            )
        return self.guard.ensure(text)

    def clarification(
        self,
        *,
        goal: str,
        options: Sequence[str],
        language: str = "en",
    ) -> str:
        if len(options) < 2:
            raise ValueError("clarification requires at least two options")
        lang = _language(language)
        choices = self._natural_join(options, language=lang)
        if lang == "zh":
            text = f"关于{goal}，可以选择{choices}。您想选哪一个？"
        elif lang == "de":
            text = (
                f"Fuer {goal} kann ich {choices} verwenden. Welche Option moechten Sie?"
            )
        else:
            text = f"For {goal}, I can use {choices}. Which option would you like?"
        return self.guard.ensure(text)

    def confirmation(
        self,
        *,
        action_bundle: Sequence[str],
        language: str = "en",
    ) -> str:
        if not action_bundle:
            raise ValueError("confirmation requires an action bundle")
        lang = _language(language)
        actions = self._natural_join(action_bundle, language=lang)
        if lang == "zh":
            text = f"这会{actions}。要我继续吗？"
        elif lang == "de":
            text = f"Dabei werde ich {actions}. Soll ich fortfahren?"
        else:
            text = f"This will {actions}. Shall I go ahead?"
        return self.guard.ensure(text)

    def partial_completion(
        self,
        *,
        completed: Sequence[str],
        blocked_goal: str,
        reason: str,
        language: str = "en",
    ) -> str:
        if not completed:
            return self.limitation(
                goal=blocked_goal,
                reason=reason,
                outcome=GateOutcome.UNSUPPORTED_CAPABILITY,
                language=language,
            )
        lang = _language(language)
        done = self._natural_join(completed, language=lang)
        if lang == "zh":
            text = (
                f"我已完成{done}，但无法{blocked_goal}，因为{reason}。后一项没有执行。"
            )
        elif lang == "de":
            text = (
                f"Ich habe {done} erledigt, konnte aber {blocked_goal} nicht ausfuehren, "
                f"weil {reason}. Diese Aktion wurde nicht ausgefuehrt."
            )
        else:
            text = (
                f"I completed {done}, but I couldn't {blocked_goal} because {reason}. "
                "That action was not performed."
            )
        return self.guard.ensure(text)

    def completion(self, summary: str, *, language: str = "en") -> str:
        lang = _language(language)
        if lang == "zh":
            text = f"已完成：{summary}。"
        elif lang == "de":
            text = f"Erledigt: {summary}."
        else:
            text = f"Done. {summary}."
        return self.guard.ensure(text)

    def range_estimate(
        self,
        *,
        initial_soc: int,
        final_soc: int,
        distance_km: str,
        language: str = "en",
    ) -> str:
        lang = _language(language)
        if lang == "zh":
            text = (
                f"我估算从百分之{initial_soc}的电量降到百分之{final_soc}时，"
                f"可行驶{distance_km}公里。"
            )
        elif lang == "de":
            text = (
                f"Ich schaetze die Reichweite von {initial_soc} Prozent bis "
                f"{final_soc} Prozent auf {distance_km} Kilometer."
            )
        else:
            text = (
                f"I estimate the driving range from {initial_soc} percent to "
                f"{final_soc} percent is {distance_km} kilometers."
            )
        return self.guard.ensure(text)

    def poi_search_results(
        self,
        *,
        location: str,
        poi_names: Sequence[str],
        poi_opening_hours: Sequence[str] | None = None,
        language: str = "en",
    ) -> str:
        if not poi_names:
            raise ValueError("POI search results require at least one name")
        lang = _language(language)
        labels: Sequence[str] = poi_names
        if poi_opening_hours is not None:
            if isinstance(poi_opening_hours, (str, bytes)) or len(
                poi_opening_hours
            ) != len(poi_names):
                raise ValueError("POI opening hours must align with every name")
            detail_guard = TTSGuard(max_characters=120)
            if any(
                not isinstance(opening_hours, str)
                or not opening_hours.strip()
                or opening_hours != opening_hours.strip()
                or detail_guard.violations(opening_hours)
                or any(
                    unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
                    for character in opening_hours
                )
                for opening_hours in poi_opening_hours
            ):
                raise ValueError("POI opening hours must be safe non-empty text")
            if lang == "zh":
                labels = tuple(
                    f"{name}，营业时间为{opening_hours}"
                    for name, opening_hours in zip(
                        poi_names, poi_opening_hours, strict=True
                    )
                )
            elif lang == "de":
                labels = tuple(
                    f"{name} mit Oeffnungszeiten {opening_hours}"
                    for name, opening_hours in zip(
                        poi_names, poi_opening_hours, strict=True
                    )
                )
            else:
                labels = tuple(
                    f"{name}, with opening hours {opening_hours}"
                    for name, opening_hours in zip(
                        poi_names, poi_opening_hours, strict=True
                    )
                )
        names = self._natural_join(labels, language=lang)
        if lang == "zh":
            text = f"我在{location}找到了{names}。您想导航到哪一个地点？"
        elif lang == "de":
            text = (
                f"Ich habe in {location} {names} gefunden. "
                "Zu welchem Ort moechtest du navigieren?"
            )
        else:
            text = (
                f"I found {names} in {location}. "
                "Which point of interest would you like directions to?"
            )
        return self.guard.ensure(text)

    def destination_route_options(
        self,
        *,
        destination: str,
        route_highlights: Sequence[tuple[str, str, str, int, bool]],
        further_route_count: int,
        language: str = "en",
        replacement_preview: bool = False,
    ) -> str:
        if not 1 <= len(route_highlights) <= 2 or further_route_count < 0:
            raise ValueError("route presentation requires alternatives")
        lang = _language(language)
        total_route_count = further_route_count + len(route_highlights)
        details: list[str] = []
        for index, (
            alias,
            via,
            distance_km,
            duration_minutes,
            includes_toll,
        ) in enumerate(route_highlights):
            duration = self._duration(duration_minutes, language=lang)
            combined = len(route_highlights) == 1
            if lang == "zh":
                kind = "最快且最短" if combined else ("最快" if index == 0 else "最短")
                toll = "，包含收费路段" if includes_toll else "，不含收费路段"
                details.append(
                    f"{kind}的是{alias}条，经由{via}，距离{distance_km}公里，"
                    f"预计{duration}{toll}。"
                )
            elif lang == "de":
                kind = (
                    "schnellste und kuerzeste"
                    if combined
                    else ("schnellste" if index == 0 else "kuerzeste")
                )
                toll = " mit Mautstrecken" if includes_toll else " ohne Mautstrecken"
                details.append(
                    f"Die {kind} ist Route {alias} ueber {via}, "
                    f"{distance_km} Kilometer und etwa {duration}{toll}."
                )
            else:
                kind = (
                    "fastest and shortest"
                    if combined
                    else ("fastest" if index == 0 else "shortest")
                )
                toll = (
                    " and includes toll roads"
                    if includes_toll
                    else " and has no toll roads"
                )
                details.append(
                    f"The {kind} is the {alias} route via {via}, at "
                    f"{distance_km} kilometers and about {duration}{toll}."
                )
        if lang == "zh":
            if further_route_count:
                follow_up = (
                    f"另有{further_route_count}条备选路线。您想了解它们的详情，"
                    "还是使用上面介绍的路线？"
                )
            else:
                follow_up = "您想使用上面哪一条路线？"
            text = (
                f"我找到了前往{destination}的{total_route_count}条路线。"
                f"{''.join(details)}"
                f"{follow_up}"
            )
        elif lang == "de":
            if further_route_count:
                follow_up = (
                    f"Es gibt {further_route_count} weitere Routen. Moechtest du "
                    "Details dazu, oder soll ich eine der beschriebenen Routen nehmen?"
                )
            else:
                follow_up = "Welche der beschriebenen Routen soll ich nehmen?"
            text = (
                f"Ich habe {total_route_count} Routen nach {destination} gefunden. "
                f"{' '.join(details)} "
                f"{follow_up}"
            )
        else:
            if replacement_preview:
                selection = "I selected the fastest route by default for this segment"
                if len(route_highlights) == 1:
                    selection += "; it is also the shortest route"
                if further_route_count:
                    alternatives = (
                        f"There are {further_route_count} further route alternatives. "
                    )
                else:
                    alternatives = ""
                follow_up = (
                    alternatives
                    + "I have not changed navigation yet. Which of the routes "
                    "described above should I use when you want me to replace the "
                    "destination?"
                )
                text = (
                    f"{selection}. I found {total_route_count} routes to "
                    f"{destination}. {' '.join(details)} {follow_up}"
                )
            elif further_route_count:
                follow_up = (
                    f"There are {further_route_count} further route alternatives. "
                    "Would you like details about them, or should I use one of the "
                    "routes described above?"
                )
            else:
                follow_up = "Which of the routes described above should I use?"
            if not replacement_preview:
                text = (
                    f"I found {total_route_count} routes to {destination}. "
                    f"{' '.join(details)} "
                    f"{follow_up}"
                )
        return self.guard.ensure(text)

    def route_alternative_details(
        self,
        *,
        route_options: Sequence[tuple[str, str, str, int, bool]],
        language: str = "en",
    ) -> str:
        if not route_options:
            raise ValueError("alternative route details require at least one route")
        lang = _language(language)
        details: list[str] = []
        for alias, via, distance_km, duration_minutes, includes_toll in route_options:
            duration = self._duration(duration_minutes, language=lang)
            if lang == "zh":
                toll = "，包含收费路段" if includes_toll else "，不含收费路段"
                details.append(
                    f"{alias}条经由{via}，距离{distance_km}公里，预计{duration}{toll}"
                )
            elif lang == "de":
                toll = " mit Mautstrecken" if includes_toll else " ohne Mautstrecken"
                details.append(
                    f"Route {alias} ueber {via}, {distance_km} Kilometer, "
                    f"etwa {duration}{toll}"
                )
            else:
                toll = " with toll roads" if includes_toll else " with no toll roads"
                details.append(
                    f"the {alias} route via {via}, {distance_km} kilometers, "
                    f"about {duration}{toll}"
                )
        options = self._natural_join(details, language=lang)
        if lang == "zh":
            text = f"我可以提供这些备选路线：{options}。您想选择哪一条？"
        elif lang == "de":
            text = (
                f"Ich kann dir diese Optionen anbieten: {options}. Welche moechtest du?"
            )
        else:
            text = f"I can offer these further options: {options}. Which route would you like?"
        return self.guard.ensure(text)

    def trip_range_assessment(
        self,
        *,
        destination: str,
        state_of_charge: str,
        available_range_km: str,
        route_distance_km: str,
        can_reach: bool,
        includes_toll: bool,
        language: str = "en",
    ) -> str:
        lang = _language(language)
        if lang == "zh":
            conclusion = (
                "无需途中充电即可到达" if can_reach else "无法在不充电的情况下到达"
            )
            charge = "途中不需要充电。" if can_reach else "途中需要充电。"
            text = (
                f"我已检查。当前电量为百分之{state_of_charge}，"
                f"可用续航为{available_range_km}公里。"
                f"前往{destination}的最快路线为{route_distance_km}公里，因此{conclusion}。{charge}"
            )
            if includes_toll:
                text += "该路线包含收费路段。"
        elif lang == "de":
            conclusion = (
                "reicht die Reichweite ohne Ladestopp aus"
                if can_reach
                else "reicht die Reichweite ohne Ladestopp nicht aus"
            )
            charge = (
                "Du musst unterwegs nicht laden."
                if can_reach
                else "Du musst unterwegs laden."
            )
            text = (
                f"Ich habe die Reichweite geprueft. Der Ladestand betraegt "
                f"{state_of_charge} Prozent und die verfuegbare "
                f"Reichweite {available_range_km} Kilometer. Die schnellste Route nach "
                f"{destination} ist {route_distance_km} Kilometer lang, daher {conclusion}. "
                f"{charge}"
            )
            if includes_toll:
                text += " Die Route enthaelt Mautstrecken."
        else:
            conclusion = (
                "you can reach it without charging"
                if can_reach
                else "you cannot reach it without charging"
            )
            charge = (
                "You do not need to charge along the way."
                if can_reach
                else "You will need to charge before or along the trip."
            )
            text = (
                f"I checked your current range. Your battery is at "
                f"{state_of_charge} percent with "
                f"{available_range_km} kilometers of available range. The fastest route "
                f"to {destination} is {route_distance_km} kilometers, so {conclusion}. "
                f"{charge}"
            )
            if includes_toll:
                text += " The route includes toll roads."
        return self.guard.ensure(text)

    def calendar_summary(self, *, entries: Sequence[str], language: str = "en") -> str:
        lang = _language(language)
        if not entries:
            if lang == "zh":
                text = "我没有查到今天的日程。"
            elif lang == "de":
                text = "Ich habe fuer heute keine Kalendereintraege gefunden."
            else:
                text = "I found no calendar entries for today."
            return self.guard.ensure(text)

        details = self._natural_join(entries, language=lang)
        if lang == "zh":
            text = f"我查到今天有{len(entries)}项日程：{details}。"
        elif lang == "de":
            text = (
                f"Ich habe fuer heute {len(entries)} Kalendereintraege gefunden: "
                f"{details}."
            )
        else:
            noun = "entry" if len(entries) == 1 else "entries"
            text = f"I found {len(entries)} calendar {noun} for today: {details}."
        return self.guard.ensure(text)

    def navigation_summary(
        self,
        *,
        active: bool,
        waypoint_names: Sequence[str] = (),
        vias: Sequence[str] = (),
        road_types: Sequence[str] = (),
        aliases: Sequence[str] = (),
        includes_toll: bool | None = None,
        distance_km: str | None = None,
        duration_minutes: int | None = None,
        details_available: bool = False,
        language: str = "en",
    ) -> str:
        lang = _language(language)
        if not active:
            if lang == "zh":
                text = "我已检查，当前没有运行导航。"
            elif lang == "de":
                text = "Ich habe nachgesehen. Die Navigation ist derzeit nicht aktiv."
            else:
                text = "I checked, and navigation is not currently running."
            return self.guard.ensure(text)

        if not details_available:
            if lang == "zh":
                text = "我已检查，导航正在运行，但无法核实路线详情。"
            elif lang == "de":
                text = (
                    "Ich habe nachgesehen. Die Navigation ist aktiv, aber ich konnte "
                    "die Routendetails nicht verifizieren."
                )
            else:
                text = (
                    "I checked, and navigation is active, but I couldn't verify the "
                    "route details."
                )
            return self.guard.ensure(text)

        path = self._route_path(waypoint_names, language=lang)
        waypoints = self._conjunctive_join(waypoint_names, language=lang)
        via = self._conjunctive_join(vias, language=lang)
        roads = self._conjunctive_join(road_types, language=lang)
        route_aliases = self._conjunctive_join(aliases, language=lang)
        duration = self._duration(duration_minutes or 0, language=lang)
        if lang == "zh":
            text = "我已检查完整路线详情，导航正在运行。"
            if path:
                text += f"路线{path}。"
            if waypoints:
                text += f"航点为{waypoints}。"
            if via:
                text += f"路线经由{via}。"
            if distance_km is not None:
                text += f"总距离为{distance_km}公里，预计用时{duration}。"
            if roads:
                text += f"道路类型为{roads}。"
            if includes_toll is not None:
                text += "包含收费道路。" if includes_toll else "不包含收费道路。"
            if route_aliases:
                text += f"路线标签为{route_aliases}。"
        elif lang == "de":
            text = "Ich habe die vollstaendigen Routendetails geprueft. Die Navigation ist aktiv."
            if path:
                text += f" Die Route fuehrt {path}."
            if waypoints:
                text += f" Die Wegpunkte sind {waypoints}."
            if via:
                text += f" Sie verlaeuft via {via}."
            if distance_km is not None:
                text += (
                    f" Die Gesamtstrecke betraegt {distance_km} Kilometer und dauert "
                    f"etwa {duration}."
                )
            if roads:
                text += f" Die Strassentypen sind {roads}."
            if includes_toll is not None:
                text += (
                    " Die Route enthaelt Mautstrassen."
                    if includes_toll
                    else " Die Route enthaelt keine Mautstrassen."
                )
            if route_aliases:
                text += f" Sie ist als {route_aliases} gekennzeichnet."
        else:
            text = "I checked the comprehensive route details. Navigation is active."
            if path:
                text += f" The route runs {path}."
            if waypoints:
                noun = "waypoint is" if len(waypoint_names) == 1 else "waypoints are"
                text += f" The {noun} {waypoints}."
            if via:
                text += f" It travels via {via}."
            if distance_km is not None:
                text += (
                    f" The total distance is {distance_km} kilometers and the estimated "
                    f"travel time is {duration}."
                )
            if roads:
                noun = "road type is" if len(road_types) == 1 else "road types are"
                text += f" The {noun} {roads}."
            if includes_toll is not None:
                text += (
                    " The route includes toll roads."
                    if includes_toll
                    else " The route has no toll roads."
                )
            if route_aliases:
                text += f" It is labeled {route_aliases}."
        return self.guard.ensure(text)

    @classmethod
    def _route_path(cls, names: Sequence[str], *, language: str) -> str:
        cleaned = [" ".join(name.split()) for name in names]
        if not cleaned:
            return ""
        if len(cleaned) == 1:
            if language == "zh":
                return f"位于{cleaned[0]}"
            if language == "de":
                return f"bei {cleaned[0]}"
            return f"at {cleaned[0]}"
        middle = cls._conjunctive_join(cleaned[1:-1], language=language)
        if language == "zh":
            through = f"，途经{middle}" if middle else ""
            return f"从{cleaned[0]}出发{through}，前往{cleaned[-1]}"
        if language == "de":
            through = f" ueber {middle}" if middle else ""
            return f"von {cleaned[0]}{through} nach {cleaned[-1]}"
        through = f" through {middle}" if middle else ""
        return f"from {cleaned[0]}{through} to {cleaned[-1]}"

    @staticmethod
    def _conjunctive_join(options: Sequence[str], *, language: str) -> str:
        cleaned = [" ".join(option.split()) for option in options]
        if not cleaned:
            return ""
        if len(cleaned) == 1:
            return cleaned[0]
        if language == "zh":
            return "、".join(cleaned[:-1]) + "和" + cleaned[-1]
        conjunction = " und " if language == "de" else " and "
        if len(cleaned) == 2:
            return conjunction.join(cleaned)
        return ", ".join(cleaned[:-1]) + "," + conjunction + cleaned[-1]

    @staticmethod
    def _duration(total_minutes: int, *, language: str) -> str:
        hours, minutes = divmod(total_minutes, 60)
        if language == "zh":
            parts = []
            if hours:
                parts.append(f"{hours}小时")
            if minutes or not parts:
                parts.append(f"{minutes}分钟")
            return "".join(parts)
        if language == "de":
            parts = []
            if hours:
                parts.append(f"{hours} Stunde" if hours == 1 else f"{hours} Stunden")
            if minutes or not parts:
                parts.append(
                    f"{minutes} Minute" if minutes == 1 else f"{minutes} Minuten"
                )
            return " und ".join(parts)
        parts = []
        if hours:
            parts.append(f"{hours} hour" if hours == 1 else f"{hours} hours")
        if minutes or not parts:
            parts.append(f"{minutes} minute" if minutes == 1 else f"{minutes} minutes")
        return " and ".join(parts)

    def bundle_completion(
        self,
        *,
        action_bundle: Sequence[Any],
        policy_operations: Sequence[str | None] = (),
        language: str = "en",
    ) -> str:
        if not action_bundle:
            raise ValueError("bundle completion requires completed actions")
        operations = list(policy_operations)
        if operations and len(operations) != len(action_bundle):
            raise ValueError("completion operations must align with actions")
        if not operations:
            operations = [None] * len(action_bundle)
        lang = _language(language)
        if lang == "zh":
            return self.guard.ensure("已完成请求的操作。")
        if lang == "de":
            return self.guard.ensure("Ich habe die angeforderten Aktionen ausgefuehrt.")
        completed = [
            self.describe_completed_call(call, operation=operation)
            for call, operation in zip(action_bundle, operations, strict=True)
        ]
        return self.guard.ensure(f"Done. I {self._and_join(completed)}.")

    def temperature_zone_difference_completion(
        self,
        completion: str,
        *,
        other_zone: str,
        other_temperature: str,
        difference_celsius: str,
        language: str = "en",
    ) -> str:
        """Append POL-012 facts after a verified single-zone temperature SET."""

        lang = _language(language)
        if lang == "zh":
            notice = (
                f"另一侧的{other_zone}区域仍为{other_temperature}摄氏度，"
                f"两侧温差为{difference_celsius}摄氏度。"
            )
        elif lang == "de":
            notice = (
                f"Die {other_zone}-Zone bleibt bei {other_temperature} Grad Celsius; "
                f"der Temperaturunterschied betraegt {difference_celsius} Grad Celsius."
            )
        else:
            notice = (
                f"The {other_zone} zone remains at {other_temperature} degrees Celsius, "
                f"so the temperature difference is {difference_celsius} degrees Celsius."
            )
        return self.guard.ensure(f"{completion} {notice}")

    def navigation_waypoint_delete_completion(
        self,
        *,
        waypoint_name: str,
        destination_name: str,
        route_via: str,
        remaining_destination_name: str | None = None,
        language: str = "en",
    ) -> str:
        lang = _language(language)
        if lang == "zh":
            text = (
                f"已从路线中移除{waypoint_name}。导航现在经由{route_via}直达"
                f"{destination_name}。"
            )
            if remaining_destination_name is not None:
                text += f"之后继续沿原路线前往{remaining_destination_name}。"
            return self.guard.ensure(text)
        if lang == "de":
            text = (
                f"Erledigt. Ich habe {waypoint_name} aus der Route entfernt. "
                f"Die Navigation fuehrt jetzt direkt ueber {route_via} nach "
                f"{destination_name}."
            )
            if remaining_destination_name is not None:
                text += (
                    " Danach folgt sie der bestehenden Route weiter nach "
                    f"{remaining_destination_name}."
                )
            return self.guard.ensure(text)
        text = (
            f"Done. I removed {waypoint_name} from the route. Navigation now "
            f"continues directly to {destination_name} via {route_via}."
        )
        if remaining_destination_name is not None:
            text += (
                f" It then follows the existing route to {remaining_destination_name}."
            )
        return self.guard.ensure(text)

    def fastest_route_default_completion(
        self,
        completion: str,
        *,
        includes_toll: bool,
        segment_only: bool = False,
        language: str = "en",
    ) -> str:
        lang = _language(language)
        if lang == "zh":
            detail = (
                "新的直连路段采用了最快路线。"
                if segment_only
                else "我为每一段都采用了最快路线。"
            )
            toll = "所选路线包含收费路段。" if includes_toll else ""
            offer = "你想了解备选路线吗？"
        elif lang == "de":
            detail = (
                "Fuer den neuen direkten Abschnitt habe ich die schnellste Route gewaehlt."
                if segment_only
                else "Ich habe fuer jeden Abschnitt die schnellste Route gewaehlt."
            )
            toll = "Die gewaehlte Route enthaelt Mautstrecken." if includes_toll else ""
            offer = "Moechtest du Informationen zu den Alternativrouten?"
        else:
            detail = (
                "I used the fastest route for the new direct segment."
                if segment_only
                else "I used the fastest route for each segment."
            )
            toll = "The selected route includes toll roads." if includes_toll else ""
            offer = "Would you like information about the alternative routes?"
        return self.guard.ensure(
            " ".join(part for part in (completion, detail, toll, offer) if part)
        )

    def route_toll_notice_completion(
        self,
        completion: str,
        *,
        includes_toll: bool,
        language: str = "en",
    ) -> str:
        if not includes_toll:
            return completion
        lang = _language(language)
        if lang == "zh":
            notice = "更新后的路线包含收费路段。"
        elif lang == "de":
            notice = "Die aktualisierte Route enthaelt Mautstrecken."
        else:
            notice = "The updated route includes toll roads."
        return self.guard.ensure(f"{completion} {notice}")

    def gate_limitation(
        self,
        *,
        goal: str,
        outcome: GateOutcome,
        language: str = "en",
    ) -> str:
        reasons = {
            GateOutcome.UNSUPPORTED_CAPABILITY: "the required control is not currently available",
            GateOutcome.UNSUPPORTED_PARAMETER: "the available control cannot express the requested setting",
            GateOutcome.UNAVAILABLE_EVIDENCE: "I couldn't verify the required current information",
            GateOutcome.POLICY_CONFLICT: "the requested action conflicts with the current safety policy",
            GateOutcome.INVALID_PROPOSAL: "I couldn't validate a safe action",
        }
        return self.limitation(
            goal=self.humanize(goal),
            reason=reasons.get(outcome, "the required information is not available"),
            outcome=outcome,
            language=language,
        )

    def prerequisite_limitation(
        self,
        *,
        goal: str,
        prerequisite: str,
        language: str = "en",
    ) -> str:
        """Name an unavailable prerequisite without exposing callable identifiers."""

        lang = _language(language)
        goal_text = self.humanize(goal)
        prerequisite_text = self.humanize(prerequisite)
        if lang == "zh":
            text = (
                f"我无法{prerequisite_text}，因为对应控制当前不可用，所以也无法安全地{goal_text}。"
                "这两项操作都没有执行。"
            )
        elif lang == "de":
            text = (
                f"Ich kann {prerequisite_text} nicht ausfuehren, weil die dafuer "
                f"erforderliche Steuerung derzeit nicht verfuegbar ist. Deshalb kann "
                f"ich {goal_text} nicht sicher ausfuehren. Keine der beiden Aktionen "
                "wurde ausgefuehrt."
            )
        else:
            text = (
                f"I can't {prerequisite_text} because that control is not currently "
                f"available, so I also can't safely {goal_text}. I haven't performed "
                "either action."
            )
        return self.guard.ensure(text)

    def relative_temperature_prerequisite_limitation(
        self,
        *,
        current_temperature: str,
        target_temperature: str,
        language: str = "en",
    ) -> str:
        """Explain a compound AC failure without implying a partial write."""

        lang = _language(language)
        if lang == "zh":
            text = (
                "我无法安全地开启空调，因为需要先关闭车窗，但当前无法使用该控制。"
                f"我也没有把全车温度从{current_temperature}度调到{target_temperature}度。"
                "请先手动关闭车窗，然后让我重试这两项操作。"
            )
        elif lang == "de":
            text = (
                "Ich konnte die Klimaanlage nicht sicher einschalten, weil zuerst die "
                "Fenster geschlossen werden muessen und diese Steuerung nicht verfuegbar "
                f"ist. Ich habe die Temperatur im gesamten Fahrzeug auch nicht von "
                f"{current_temperature} auf {target_temperature} Grad geaendert. Bitte "
                "schliesse die Fenster manuell und bitte mich dann, beide Aenderungen "
                "erneut auszufuehren."
            )
        else:
            text = (
                "I couldn't safely turn on the air conditioning because the windows "
                "must be closed first and that control isn't available. I also did not "
                f"change the whole-car temperature from {current_temperature} to "
                f"{target_temperature} degrees. Please close the windows manually, then "
                "ask me to try both changes again."
            )
        return self.guard.ensure(text)

    @staticmethod
    def humanize(value: str) -> str:
        aliases = {
            "set_air_conditioning": "turn on the air conditioning",
            "enable_air_conditioning": "turn on the air conditioning",
            "disable_air_conditioning": "turn off the air conditioning",
            "set_fan_speed": "set the fan speed",
            "set_fan_airflow_direction": "set the fan airflow direction",
            "set_window_position": "move the window",
            "close_driver_window_for_ac": "close the driver window",
            "close_passenger_window_for_ac": "close the passenger window",
            "close_driver_rear_window_for_ac": "close the driver-side rear window",
            "close_passenger_rear_window_for_ac": (
                "close the passenger-side rear window"
            ),
            "set_minimum_fan_for_ac": "set the minimum fan speed",
            "open_sunroof": "open the sunroof",
            "set_sunroof_position": "move the sunroof",
            "open_sunshade_for_sunroof": "open the sunshade",
            "set_fog_lights": "change the fog lights",
            "set_high_beams": "change the high beams",
            "set_low_beams": "change the low beams",
            "start_navigation": "start navigation",
            "set_new_navigation": "start navigation",
            "call_phone_by_number": "place the phone call",
            "send_email": "send the email",
        }
        return aliases.get(value, value.replace("_", " "))

    def describe_call(self, call: Any) -> str:
        if hasattr(call, "tool_name"):
            name = call.tool_name
            arguments = call.arguments
        else:
            name = call.get("tool_name", "")
            arguments = call.get("arguments", {})
        action = self.humanize(str(name))
        if not arguments:
            return action
        details = ", ".join(
            f"{self.humanize(str(key))} {value}" for key, value in arguments.items()
        )
        return f"{action} with {details}"

    def describe_completed_call(
        self, call: Any, *, operation: str | None = None
    ) -> str:
        if hasattr(call, "tool_name"):
            name = str(call.tool_name)
            arguments = dict(call.arguments)
        else:
            name = str(call.get("tool_name", ""))
            arguments = dict(call.get("arguments", {}))

        if name in {"open_close_sunroof", "open_close_sunshade"}:
            target = "the sunroof" if name.endswith("sunroof") else "the sunshade"
            return self._completed_position(target, arguments.get("percentage"))
        if name == "open_close_window":
            window = self._friendly_value(arguments.get("window", ""))
            target = "all windows" if window == "all" else f"the {window} window"
            return self._completed_position(target, arguments.get("percentage"))
        if name == "open_close_trunk_door":
            action = self._friendly_value(arguments.get("action", ""))
            if action in {"open", "opened"}:
                return "opened the trunk"
            if action in {"close", "closed"}:
                return "closed the trunk"
        toggles = {
            "set_air_conditioning": "the air conditioning",
            "set_fog_lights": "the fog lights",
            "set_head_lights_high_beams": "the high beams",
            "set_head_lights_low_beams": "the low beams",
        }
        if name in toggles and isinstance(arguments.get("on"), bool):
            verb = "turned on" if arguments["on"] else "turned off"
            return f"{verb} {toggles[name]}"
        if name == "set_fan_speed":
            return f"set the fan speed to level {self._friendly_value(arguments.get('level'))}"
        if name == "set_climate_temperature":
            zone = self._friendly_value(arguments.get("seat_zone", "cabin"))
            value = self._friendly_value(arguments.get("temperature"))
            return f"set the {zone} temperature to {value} degrees Celsius"
        if name == "set_fan_airflow_direction":
            value = self._friendly_value(arguments.get("direction"))
            return f"set the fan airflow direction to {value}"
        if name == "set_air_circulation":
            value = self._friendly_value(arguments.get("mode"))
            return f"set the air circulation to {value}"
        if name == "set_window_defrost":
            window = self._friendly_value(arguments.get("window", "window"))
            enabled = arguments.get("on")
            verb = "turned on" if enabled is True else "turned off"
            return f"{verb} the {window} defroster"
        if name == "set_seat_heating":
            zone = self._friendly_value(arguments.get("seat_zone", "seat"))
            level = self._friendly_value(arguments.get("level"))
            return f"set the {zone} seat heating to level {level}"
        if name == "set_steering_wheel_heating":
            level = self._friendly_value(arguments.get("level"))
            return f"set the steering wheel heating to level {level}"
        if name == "set_reading_light":
            position = self._friendly_value(arguments.get("position", "reading"))
            verb = "turned on" if arguments.get("on") is True else "turned off"
            return f"{verb} the {position} reading light"
        if name == "set_ambient_lights":
            verb = "turned on" if arguments.get("on") is True else "turned off"
            color = self._friendly_value(arguments.get("lightcolor", "ambient"))
            return f"{verb} the {color} ambient lights"
        fixed_actions = {
            "set_new_navigation": "started navigation",
            "navigation_add_one_waypoint": "added the waypoint to the route",
            "navigation_replace_one_waypoint": "replaced the route waypoint",
            "navigation_replace_final_destination": "replaced the destination",
            "navigation_delete_waypoint": "removed the waypoint from the route",
            "navigation_delete_destination": "removed the destination",
            "delete_current_navigation": "stopped navigation",
            "call_phone_by_number": "placed the phone call",
            "send_email": "sent the email",
        }
        if name in fixed_actions:
            return fixed_actions[name]
        action = self.humanize(operation or name)
        return f"completed the requested {action} action"

    @classmethod
    def _completed_position(cls, target: str, percentage: Any) -> str:
        if isinstance(percentage, (int, float)) and not isinstance(percentage, bool):
            value = cls._friendly_value(percentage)
            if float(percentage) == 0:
                return f"closed {target}"
            return f"opened {target} to {value} percent"
        return f"moved {target} to the requested position"

    @staticmethod
    def _friendly_value(value: Any) -> str:
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value).casefold().replace("_", " ")

    @staticmethod
    def _and_join(options: Sequence[str]) -> str:
        cleaned = [" ".join(option.split()) for option in options]
        if len(cleaned) == 1:
            return cleaned[0]
        if len(cleaned) == 2:
            return " and ".join(cleaned)
        return ", ".join(cleaned[:-1]) + ", and " + cleaned[-1]

    @staticmethod
    def _natural_join(options: Sequence[str], *, language: str) -> str:
        cleaned = [" ".join(option.split()) for option in options]
        if len(cleaned) == 1:
            return cleaned[0]
        conjunction = (
            "或" if language == "zh" else (" oder " if language == "de" else " or ")
        )
        if language == "zh":
            return "、".join(cleaned[:-1]) + conjunction + cleaned[-1]
        if len(cleaned) == 2:
            return conjunction.join(cleaned)
        return ", ".join(cleaned[:-1]) + "," + conjunction + cleaned[-1]
