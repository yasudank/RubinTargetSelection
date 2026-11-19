#!/usr/bin/env python
"""
Fink Data Transfer ポータル (https://lsst.fink-portal.org/download) にジョブを投入し、トピック名を出力する。
../fink/datatransfer.py を元にしたもの。Playwright が必要なので ../fink/bin/python で実行する。

   ../fink/bin/python fink_submit.py datatransfer.yml [--shot_dir logs/daily/2026-11-05]

標準出力の最後の行にトピック名 (ftransfer_lsst_YYYY-MM-DD_NNNNNN) を出す。
終了コード:
  0  投入してジョブが終了した (-Livy- End. まで確認)
  3  ポータルの推定アラート数が 0 (投入しない)
  1  それ以外の失敗 (shot_dir にスクリーンショットとページの文字列を残す)

注意: ポータルの推定アラート数は日付だけで決まる全体の数で、yml の extra_cond (領域制限) は反映されない。
"""

import argparse
import asyncio
import os
import re
import sys
import time

from playwright.async_api import async_playwright

URL = "https://lsst.fink-portal.org/download"
TOPIC_RE = re.compile(r"Data available at topic:\s*(ftransfer_\S+)")
TOPIC_NAME_RE = re.compile(r"Your topic name is:\s*(ftransfer_\S+)")
COUNT_RE = re.compile(r"([\d,]+)\s+alerts\b")


def log(msg):
    print(f"[fink_submit {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


async def dump(page, shot_dir, tag):
    if not shot_dir:
        return
    os.makedirs(shot_dir, exist_ok=True)
    try:
        await page.screenshot(path=os.path.join(shot_dir, f"portal_{tag}.png"), full_page=True)
        with open(os.path.join(shot_dir, f"portal_{tag}.txt"), "w") as f:
            f.write(await page.locator("body").inner_text())
    except Exception as e:  # スクリーンショットの失敗で本来のエラーを隠さない
        log(f"dump failed: {e}")


async def run(yml, shot_dir, job_timeout, headed):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=not headed)
        page = await browser.new_page(viewport={"width": 1280, "height": 2400})
        try:
            log(f"open {URL}")
            await page.goto(URL, timeout=90_000)
            await page.wait_for_selector('input[type="file"]', state="attached", timeout=90_000)

            log(f"upload {yml}")
            await page.set_input_files('input[type="file"]', yml)
            btn = page.locator("#submit_datatransfer")
            await btn.wait_for(state="visible", timeout=90_000)
            await page.wait_for_function("document.body.innerText.includes('Previous configuration loaded')",
                                         timeout=60_000)
            await page.wait_for_timeout(3000)  # 推定アラート数の更新を待つ

            body = await page.locator("body").inner_text()
            m = COUNT_RE.search(body)
            n_est = int(m.group(1).replace(",", "")) if m else None
            log(f"portal estimate (whole night, no sky cut): {n_est} alerts")
            if n_est == 0:
                await dump(page, shot_dir, "zero")
                return 3

            if not await btn.is_enabled():
                raise RuntimeError("Submit job ボタンが有効になりません")
            await btn.click()
            log("submitted; waiting for '-Livy- End.'")

            t0 = time.time()
            while True:
                body = await page.locator("body").inner_text()
                if "-Livy- End." in body:
                    break
                if re.search(r"-Livy-.*(Error|error|ERROR|Exception|failed)", body):
                    raise RuntimeError("ポータルのログにエラーがあります")
                if time.time() - t0 > job_timeout:
                    raise RuntimeError(f"{job_timeout} 秒たってもジョブが終わりません")
                await page.wait_for_timeout(5000)
            log(f"job finished in {time.time() - t0:.0f} s")
            await dump(page, shot_dir, "done")

            m = TOPIC_RE.search(body) or TOPIC_NAME_RE.search(body)
            if not m:
                raise RuntimeError("トピック名が見つかりません")
            print(m.group(1), flush=True)
            return 0
        except Exception as e:
            log(f"ERROR: {e}")
            await dump(page, shot_dir, "error")
            return 1
        finally:
            await browser.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("yml")
    ap.add_argument("--shot_dir", default=None, help="スクリーンショットとページの文字列を保存するディレクトリ")
    ap.add_argument("--job_timeout", type=int, default=1800, help="ジョブ終了を待つ最大秒数 (既定 1800)")
    ap.add_argument("--headed", action="store_true", help="ブラウザを表示する (デバッグ用)")
    a = ap.parse_args()
    if not os.path.exists(a.yml):
        sys.exit(f"not found: {a.yml}")
    sys.exit(asyncio.run(run(os.path.abspath(a.yml), a.shot_dir, a.job_timeout, a.headed)))


if __name__ == "__main__":
    main()
